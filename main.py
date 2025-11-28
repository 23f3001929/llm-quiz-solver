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

# Chat Proxy (Audio proxy bypassed via Google Speech)
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
    """Bypasses AI Pipe proxy using Google Speech Recognition."""
    print(f"    [Tool] 🔊 Found Audio: {url}")
    try:
        resp = requests.get(url, timeout=30)
        if 'text/html' in resp.headers.get('Content-Type', ''): return ""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save and Convert
            orig_path = os.path.join(temp_dir, "original")
            with open(orig_path, "wb") as f: f.write(resp.content)
            
            wav_path = os.path.join(temp_dir, "converted.wav")
            AudioSegment.from_file(orig_path).export(wav_path, format="wav")
            
            # Transcribe
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
                text = r.recognize_google(audio)
            
            print(f"    [Tool] Transcript: {text[:50]}...")
            return f"\n[AUDIO TRANSCRIPT]: \"{text}\"\n"
    except Exception as e:
        print(f"    [Tool] Audio Error: {e}")
        return ""

async def process_csv_url(url: str) -> str:
    print(f"    [Tool] 📊 Found CSV: {url}")
    try:
        r = requests.get(url, timeout=15)
        if 'text/html' in r.headers.get('Content-Type', ''): return ""
        content = r.content.decode("utf-8", errors="replace")
        return content
    except:
        return ""

def get_csv_stats(text: str) -> str:
    """Pre-calculates math so the AI doesn't have to guess."""
    if not text: return ""
    # Extract all numbers
    nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', text.replace(',', ''))]
    if not nums: return ""
    
    total = sum(nums)
    count = len(nums)
    maximum = max(nums)
    
    # Format integers nicely
    if total.is_integer(): total = int(total)
    if maximum.is_integer(): maximum = int(maximum)
    
    return f"""
    [SYSTEM MATH HINTS]:
    - Count of numbers found: {count}
    - Sum of numbers: {total}
    - Max value: {maximum}
    """

def extract_secret_regex(text: str) -> str:
    """Finds codes like 'The secret is XYZ'."""
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

            # 1. FIND LINKS
            links = await page.evaluate("""() => {
                let arr = [];
                document.querySelectorAll('a').forEach(a => arr.push(a.href));
                document.querySelectorAll('audio, source').forEach(el => arr.push(el.src));
                return arr;
            }""")

            # 2. PROCESS FILES
            evidence = ""
            csv_content = ""
            seen = set()
            
            for link in links:
                if link in seen or link.rstrip('/') == task_url.rstrip('/'): continue
                seen.add(link)
                
                # Audio
                if any(x in link.lower() for x in ['.mp3', '.wav', '.opus', 'audio']):
                    evidence += await process_audio_google(link)
                
                # CSV
                elif 'csv' in link.lower() or 'download' in link.lower():
                    csv_data = await process_csv_url(link)
                    if csv_data:
                        csv_content = csv_data
                        evidence += f"\n[CSV FILE CONTENT (Truncated)]:\n{csv_data[:500]}\n"

            # 3. PREPARE INTELLIGENCE
            regex_secret = extract_secret_regex(page_text)
            math_hints = get_csv_stats(csv_content)

            # 4. AI PROMPT
            prompt = f"""
            You are a Quiz Solving Robot.
            
            === INTERNAL DATA ===
            EMAIL: "{email}"
            SECRET: "{student_secret}"
            
            === PAGE DATA ===
            TEXT: {page_text}
            FILES FOUND: {evidence}
            MATH ANALYSIS: {math_hints}
            REGEX MATCH: {regex_secret if regex_secret else "None"}
            
            === INSTRUCTIONS ===
            1. SCRAPING: If the page text says "The secret is XYZ", output "XYZ". Do NOT output placeholders like "<CODE>". If Regex Match found something, prefer that.
            2. MATH: If Audio says "Sum the numbers", use the "Sum" from MATH ANALYSIS.
            3. IDENTITY: If asked for "your secret", use INTERNAL DATA SECRET.
            
            Return JSON:
            {{ "submission_url": "...", "payload": {{ "email": "{email}", "secret": "{student_secret}", "url": "{task_url}", "answer": <REAL VALUE> }} }}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            submission_url = data.get("submission_url")
            payload = data.get("payload")

            # --- POST-PROCESSING & SAFETY NETS ---
            
            # URL Fix
            if submission_url and not submission_url.startswith("http"):
                submission_url = urljoin(task_url, submission_url)
            
            # Answer Cleanup (Fixes Level 2 Hallucination)
            ans = str(payload.get("answer", ""))
            if "<" in ans and ">" in ans: # Detects placeholders like <CODE>
                if regex_secret:
                    print("    ⚠️ Detected placeholder. Swapping with Regex match.")
                    payload["answer"] = regex_secret
            
            # Identity Swap (Fixes Level 2 Secret Mismatch)
            if "your secret" in ans.lower():
                 payload["answer"] = student_secret

            print(f"    Final Answer: {payload.get('answer')}")
            print(f"    Submitting to: {submission_url}")

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