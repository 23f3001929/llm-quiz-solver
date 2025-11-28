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

# We still use the proxy for Chat, but NOT for Audio
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
    This Bypasses the AI Pipe Proxy entirely to avoid 400 Errors.
    """
    print(f"    [Tool] 🔊 Found Audio: {url}")
    try:
        # 1. Download
        resp = requests.get(url, timeout=30)
        if 'text/html' in resp.headers.get('Content-Type', ''): return ""
        
        # 2. Convert to WAV (Google requires WAV)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save original file (likely .opus or .mp3)
            orig_path = os.path.join(temp_dir, "original_audio")
            with open(orig_path, "wb") as f:
                f.write(resp.content)
            
            # Convert to WAV using pydub + ffmpeg
            wav_path = os.path.join(temp_dir, "converted.wav")
            audio = AudioSegment.from_file(orig_path)
            audio.export(wav_path, format="wav")
            
            # 3. Transcribe with Google Web Speech API (Free)
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                
            print(f"    [Tool] Google Transcription: {text[:50]}...")
            return f"\n[AUDIO TRANSCRIPT]\n{text}\n"

    except Exception as e:
        print(f"    [Tool] Audio Error: {e}")
        return ""

async def process_csv_url(url: str) -> str:
    print(f"    [Tool] 📊 Found CSV: {url}")
    try:
        r = requests.get(url, timeout=15)
        if 'text/html' in r.headers.get('Content-Type', ''): return ""
        content = r.content.decode("utf-8", errors="replace")
        return f"\n[CSV CONTENT]\n{content}\n"
    except Exception as e:
        return ""

def extract_secret_from_page(text: str) -> str:
    """
    Uses Regex to find codes like 'The secret is 1234' if the AI is too lazy to look.
    """
    patterns = [
        r"secret\s+(?:code\s+)?(?:is|:)\s*([A-Za-z0-9]+)", # "secret is XYZ"
        r"code\s+(?:is|:)\s*([A-Za-z0-9]+)",               # "code is XYZ"
        r"cutoff\s+(?:is|:)\s*(\d+)"                       # "cutoff is 123"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def extract_numbers(text: str):
    """Finds all numbers in text for math summing."""
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text.replace(',', ''))]

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

            # Get Links directly from DOM
            media_links = await page.evaluate("""() => {
                let links = [];
                document.querySelectorAll('a').forEach(a => { if (a.href) links.push(a.href) });
                document.querySelectorAll('audio, source').forEach(el => { if (el.src) links.push(el.src) });
                return links;
            }""")

            evidence = ""
            csv_nums = []
            
            # Download Loop
            seen = set()
            for url in media_links:
                if url in seen or url.rstrip('/') == task_url.rstrip('/'): continue
                seen.add(url)
                
                # Check for Audio
                if any(x in url.lower() for x in ['.mp3', '.wav', '.opus', 'audio']):
                    evidence += await process_audio_google(url)
                
                # Check for CSV
                elif 'csv' in url.lower() or 'download' in url.lower():
                    csv_text = await process_csv_url(url)
                    if csv_text:
                        evidence += csv_text
                        csv_nums = extract_numbers(csv_text)

            # --- SMART LOGIC ---
            
            # 1. Regex Search (Fixes Level 2 "Lazy AI")
            extracted_code = extract_secret_from_page(page_text)
            
            # 2. Math Calculation (Fixes Level 3 "Wrong Sum")
            # If CSV exists, sum it.
            math_answer = None
            if csv_nums:
                math_answer = sum(csv_nums)
                # Convert to int if it's a whole number (e.g., 50.0 -> 50)
                if math_answer.is_integer(): math_answer = int(math_answer)
            
            # 3. AI Prompt
            prompt = f"""
            You are a Quiz Solver.
            
            INTERNAL CREDENTIALS:
            - EMAIL: "{email}"
            - SECRET: "{student_secret}"
            
            AUTO-DETECTED CLUES:
            - Found on Page: {extracted_code if extracted_code else "None"}
            - Calculated CSV Sum: {math_answer if math_answer else "None"}
            
            EVIDENCE:
            {evidence}
            
            PAGE TEXT:
            {page_text}
            
            TASK: 
            1. Find submission URL.
            2. Answer the question.
            
            RULES:
            - IDENTITY: If asked for "your secret/email", use INTERNAL CREDENTIALS.
            - SCRAPING: If asked for a specific code on the page, use the 'Found on Page' value if valid, or find it in the text.
            - MATH: If asked to sum numbers, prefer the 'Calculated CSV Sum'.
            
            Return JSON:
            {{ "submission_url": "...", "payload": {{ "email": "{email}", "secret": "{student_secret}", "url": "{task_url}", "answer": <value> }} }}
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
            
            # Safety Net: If AI missed the regex code, force it
            if extracted_code and "your secret" in str(payload.get("answer", "")).lower():
                payload["answer"] = extracted_code

            # Safety Net: If AI missed the credentials
            if "your secret" in str(payload.get("answer", "")).lower():
                # Check context to see if we should swap
                if "secret" in page_text.lower() and "input" in page_text.lower():
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
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}