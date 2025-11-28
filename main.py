import os
import json
import re
import asyncio
import requests
import tempfile
import time
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from openai import OpenAI
import speech_recognition as sr
from pydub import AudioSegment

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
MY_SECRET = os.getenv("MY_SECRET")

# Proxy for Chat (still needed), but we bypass it for Audio
client = OpenAI(api_key=AIPIPE_TOKEN, base_url="https://aipipe.org/openai/v1")
app = FastAPI()

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# ==========================================
# TOOLKIT
# ==========================================

async def process_audio_google(url: str) -> str:
    """
    Downloads audio, converts to WAV, and uses Google Speech Recognition.
    """
    print(f"    [Tool] 🔊 Found Audio: {url}")
    try:
        # 1. Download with increased timeout
        resp = requests.get(url, timeout=45)
        if 'text/html' in resp.headers.get('Content-Type', ''): 
            print("    [Tool] Skipping: URL returned HTML.")
            return ""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save original
            orig_path = os.path.join(temp_dir, "original_audio")
            with open(orig_path, "wb") as f:
                f.write(resp.content)
            
            # Convert to WAV (Google requirement)
            wav_path = os.path.join(temp_dir, "converted.wav")
            
            # Use pydub to convert (requires ffmpeg installed in Docker)
            audio = AudioSegment.from_file(orig_path)
            audio.export(wav_path, format="wav")
            
            # Transcribe
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                # Google Web Speech API (High quality, free)
                text = recognizer.recognize_google(audio_data)
                
            print(f"    [Tool] Transcription: {text[:50]}...")
            return f"\n[AUDIO TRANSCRIPT SOURCE: {url}]\n{text}\n"

    except Exception as e:
        print(f"    [Tool] Audio Error: {e}")
        return f"\n[AUDIO ERROR]: Could not transcribe {url} ({str(e)})\n"

async def process_csv_url(url: str) -> str:
    print(f"    [Tool] 📊 Found CSV: {url}")
    try:
        r = requests.get(url, timeout=30)
        if 'text/html' in r.headers.get('Content-Type', ''): return ""
        content = r.content.decode("utf-8", errors="replace")
        return f"\n[CSV CONTENT SOURCE: {url}]\n{content}\n"
    except Exception as e:
        return ""

def extract_numbers(text: str):
    """Helper to extract numbers for local math verification"""
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text.replace(',', ''))]

def extract_secret_regex(text: str) -> str:
    """Finds 'secret is XYZ' in text."""
    patterns = [
        r"secret\s+(?:code\s+)?(?:is|:)\s*([A-Za-z0-9]+)",
        r"code\s+(?:is|:)\s*([A-Za-z0-9]+)",
        r"cutoff\s+(?:is|:)\s*(\d+)"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None

# ==========================================
# MAIN LOGIC
# ==========================================
async def solve_quiz_task(task_url: str, email: str, student_secret: str):
    if not task_url: return
    print(f"\n[+] Processing Task: {task_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(task_url)
            await page.wait_for_selector("body", timeout=15000)
            page_text = await page.evaluate("document.body.innerText")

            # 1. EXTRACT LINKS (Raw DOM extraction)
            media_links = await page.evaluate("""() => {
                let links = [];
                document.querySelectorAll('a').forEach(a => { if (a.href) links.push(a.href) });
                document.querySelectorAll('audio, source').forEach(el => { if (el.src) links.push(el.src) });
                return links;
            }""")

            evidence = ""
            csv_nums = []
            
            # 2. INTELLIGENT DOWNLOADER
            # Strictly checks extensions to avoid mixing up CSVs and Audio
            seen = set()
            for url in media_links:
                if url in seen or url.rstrip('/') == task_url.rstrip('/'): continue
                seen.add(url)
                
                u_low = url.lower()
                
                # Check CSV first
                if u_low.endswith('.csv') or ('csv' in u_low and 'download' in u_low and not 'audio' in u_low):
                    csv_text = await process_csv_url(url)
                    if csv_text:
                        evidence += csv_text
                        csv_nums = extract_numbers(csv_text)
                
                # Check Audio second
                elif u_low.endswith(('.mp3', '.wav', '.opus', '.m4a', '.oga')) or 'audio' in u_low:
                    # Logic: Only treat as audio if it DOESN'T look like a CSV url
                    if not u_low.endswith('.csv'):
                        evidence += await process_audio_google(url)

            # 3. REASONING
            extracted_code = extract_secret_regex(page_text)
            
            # Pre-calculate Math
            math_answer = None
            if csv_nums:
                math_answer = sum(csv_nums)
                if math_answer.is_integer(): math_answer = int(math_answer)
            
            prompt = f"""
            You are an autonomous Quiz Solving Agent.
            
            === INTERNAL IDENTITY ===
            - EMAIL: "{email}"
            - SECRET: "{student_secret}"
            (Use these ONLY if the page specifically asks for "YOUR" credentials)
            
            === COLLECTED EVIDENCE ===
            {evidence}
            
            === PAGE TEXT ===
            {page_text}
            
            === CLUES DETECTED ===
            - Regex Secret Found: {extracted_code if extracted_code else "None"}
            - CSV Sum Calculated: {math_answer if math_answer else "None"}
            
            === RULES ===
            1. IDENTITY: If page asks for "your secret", "password", or "email", use INTERNAL IDENTITY.
            2. CODE: If page says "The secret code is...", use that code.
            3. MATH: If page asks to sum numbers, prefer the 'CSV Sum Calculated'.
            
            Return JSON ONLY:
            {{
                "submission_url": "...",
                "payload": {{
                    "email": "{email}",
                    "secret": "{student_secret}",
                    "url": "{task_url}",
                    "answer": <THE_FINAL_ANSWER>
                }}
            }}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            ai_data = json.loads(response.choices[0].message.content)
            submission_url = ai_data.get("submission_url")
            payload = ai_data.get("payload")

            # Final Cleanup
            if submission_url and not submission_url.startswith("http"):
                submission_url = urljoin(task_url, submission_url)
            
            # Local Logic Overrides
            # 1. Force Regex Secret if available
            if extracted_code and "your secret" in str(payload.get("answer", "")).lower():
                print("    ⚠️ Overriding with Regex Secret.")
                payload["answer"] = extracted_code

            # 2. Force Math Sum if available and answer looks wrong
            if math_answer is not None and (payload.get("answer") == 0 or payload.get("answer") == "0"):
                 print("    ⚠️ Overriding with CSV Sum.")
                 payload["answer"] = math_answer

            # 3. Force Internal Secret if needed
            if "your secret" in str(payload.get("answer", "")).lower():
                if "secret" in page_text.lower() and "input" in page_text.lower():
                    print("    ⚠️ Overriding with Internal Secret.")
                    payload["answer"] = student_secret

            print(f"    AI Answer: {payload.get('answer')}")
            print(f"    Submitting to: {submission_url}")

            if submission_url:
                res = requests.post(submission_url, json=payload).json()
                print("    Server:", res)
                
                # RECURSION
                if res.get("url"):
                    await browser.close()
                    await solve_quiz_task(res["url"], email, student_secret)

        except Exception as e:
            print("    Error:", e)
        
        if browser.is_connected():
            await browser.close()

@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}