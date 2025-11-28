import os
import json
import re
import asyncio
import requests
import tempfile
from urllib.parse import urljoin

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

# Use proxy for Chat, but we bypass it for Audio using Google
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
    Downloads audio, converts to WAV, and transcribes.
    Bypasses the broken AI Pipe proxy by using Google Speech Rec.
    """
    print(f"    [Tool] 🔊 Processing Audio: {url}")
    try:
        # Check headers first to ensure it's actually audio
        head = requests.head(url, timeout=5)
        if 'text/html' in head.headers.get('Content-Type', ''):
            print("    [Skip] URL is a webpage, not audio.")
            return ""

        resp = requests.get(url, timeout=30)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save original
            orig_path = os.path.join(temp_dir, "original_audio")
            with open(orig_path, "wb") as f: f.write(resp.content)
            
            # Convert to WAV (Requires ffmpeg in Dockerfile)
            wav_path = os.path.join(temp_dir, "converted.wav")
            try:
                AudioSegment.from_file(orig_path).export(wav_path, format="wav")
            except Exception:
                return "\n[ERROR] Audio conversion failed. Is ffmpeg installed?\n"
            
            # Transcribe
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
                text = r.recognize_google(audio)
            
            print(f"    [Tool] Transcript: {text[:50]}...")
            return f"\n[AUDIO TRANSCRIPT SOURCE: {url}]\nContent: \"{text}\"\n(Use this instruction to process the CSV data)\n"

    except Exception as e:
        print(f"    [Tool] Audio Error: {e}")
        return ""

async def process_csv_url(url: str) -> str:
    print(f"    [Tool] 📊 Processing CSV: {url}")
    try:
        # Check headers
        head = requests.head(url, timeout=5)
        if 'text/html' in head.headers.get('Content-Type', ''): return ""

        r = requests.get(url, timeout=15)
        content = r.content.decode("utf-8", errors="replace")
        
        # Limit content size to prevent token overflow (first 500 lines or 15k chars)
        lines = content.splitlines()
        if len(lines) > 500:
            content = "\n".join(lines[:500]) + "\n...[Truncated]..."
            
        return f"\n[CSV FILE CONTENT SOURCE: {url}]\n{content}\n"
    except Exception as e:
        return ""

def extract_secret_regex(text: str) -> str:
    """Backup method to find secrets if AI misses them."""
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

            # 1. EXTRACT ALL RELEVANT LINKS
            media_links = await page.evaluate("""() => {
                let links = [];
                document.querySelectorAll('a').forEach(a => { if (a.href) links.push(a.href) });
                document.querySelectorAll('audio, source').forEach(el => { if (el.src) links.push(el.src) });
                return links;
            }""")

            evidence = ""
            seen = set()
            
            # 2. SMART DOWNLOAD LOOP
            for url in media_links:
                if url in seen or url.rstrip('/') == task_url.rstrip('/'): continue
                seen.add(url)
                
                u_low = url.lower()
                
                # STRICT TYPE CHECKING to prevent "Audio Error" on CSV files
                is_audio_ext = u_low.endswith(('.mp3', '.wav', '.opus', '.m4a', '.oga'))
                is_csv_ext = u_low.endswith('.csv')
                
                if is_csv_ext:
                    evidence += await process_csv_url(url)
                elif is_audio_ext:
                    evidence += await process_audio_google(url)
                elif 'download' in u_low and not is_audio_ext:
                    # Fallback for generic download links - try CSV first
                    evidence += await process_csv_url(url)

            # 3. REASONING AGENT
            # We explicitly tell the AI to prioritize Audio Instructions over simple math.
            regex_match = extract_secret_regex(page_text)
            
            prompt = f"""
            You are an expert Data Processing Agent.
            
            === CREDENTIALS ===
            EMAIL: "{email}"
            SECRET: "{student_secret}"
            
            === EVIDENCE COLLECTED ===
            {evidence}
            
            === PAGE TEXT ===
            {page_text}
            
            === INSTRUCTIONS ===
            1. CHECK AUDIO: Read the [AUDIO TRANSCRIPT] carefully. It often contains a FILTER condition (e.g., "only sum numbers > 50000").
            2. CHECK CSV: Apply the filter from the audio to the [CSV FILE CONTENT] and perform the calculation.
            3. CHECK SECRET: If the page asks for "your secret", use the CREDENTIALS. If it asks for a "scraped code", use the text found on the page.
            
            === REGEX HINT ===
            (Use this if you can't find a code in the text): {regex_match if regex_match else "None"}
            
            Return VALID JSON ONLY:
            {{
                "submission_url": "...",
                "payload": {{
                    "email": "{email}",
                    "secret": "{student_secret}",
                    "url": "{task_url}",
                    "answer": <CALCULATED_VALUE_OR_STRING>
                }}
            }}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            submission_url = data.get("submission_url")
            payload = data.get("payload")

            # --- SAFETY NETS ---
            if submission_url and not submission_url.startswith("http"):
                submission_url = urljoin(task_url, submission_url)
            
            # Prevent "placeholder" answers
            ans_str = str(payload.get("answer", "")).lower()
            if "scraped" in ans_str or "code" in ans_str or "<" in ans_str:
                if regex_match:
                    print("    ⚠️ Replaced placeholder with Regex Match.")
                    payload["answer"] = regex_match
                elif "secret" in page_text.lower():
                    payload["answer"] = student_secret

            # Fix common "Your Secret" mistake
            if "your secret" in ans_str:
                 payload["answer"] = student_secret

            print(f"    AI Answer: {payload.get('answer')}")
            print(f"    Submitting to: {submission_url}")

            # 4. SUBMIT & RECURSE
            if submission_url:
                res = requests.post(submission_url, json=payload).json()
                print("    Server:", res)
                
                if res.get("url"):
                    await browser.close()
                    await solve_quiz_task(res["url"], email, student_secret)

        except Exception as e:
            print("    Error:", e)
        
        if browser.is_connected():
            await browser.close()

@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}