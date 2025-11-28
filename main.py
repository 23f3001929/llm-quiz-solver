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
        resp = requests.get(url, timeout=30)
        if 'text/html' in resp.headers.get('Content-Type', ''): return ""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save original
            orig_path = os.path.join(temp_dir, "original_audio")
            with open(orig_path, "wb") as f:
                f.write(resp.content)
            
            # Convert to WAV (Google requirement)
            wav_path = os.path.join(temp_dir, "converted.wav")
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
        return f"\n[AUDIO ERROR]: Could not transcribe {url}\n"

async def process_csv_url(url: str) -> str:
    print(f"    [Tool] 📊 Found CSV: {url}")
    try:
        r = requests.get(url, timeout=15)
        if 'text/html' in r.headers.get('Content-Type', ''): return ""
        content = r.content.decode("utf-8", errors="replace")
        return f"\n[CSV CONTENT SOURCE: {url}]\n{content}\n"
    except Exception as e:
        return ""

def extract_numbers(text: str):
    """Helper to extract numbers for local math verification"""
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
                elif u_low.endswith(('.mp3', '.wav', '.opus', '.m4a')) or 'audio' in u_low:
                    # Logic: Only treat as audio if it DOESN'T look like a CSV url
                    if not u_low.endswith('.csv'):
                        evidence += await process_audio_google(url)

            # 3. THE REASONING BRAIN
            # We use "Chain of Thought" - asking the AI to explain its logic first
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
            
            === YOUR MISSION ===
            1. ANALYZE: Look at the Audio Transcripts (if any). Do they give instructions? (e.g. "Sum the numbers").
            2. LOOKUP: Look at the CSV Content. Apply the instructions to this data.
            3. IDENTIFY: Does the page text provide a specific code? Or does it ask for your Identity?
            
            === RULES ===
            - If Audio says "The code is...", use that code.
            - If Audio says "Sum the numbers...", calculate the sum from the CSV data.
            - If Page says "Enter your secret", use the INTERNAL IDENTITY SECRET.
            - If Page says "The secret code is 123", use "123".
            
            Return JSON ONLY:
            {{
                "thought_process": "Briefly explain how you found the answer here...",
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
            thought = ai_data.get("thought_process", "No thought provided")

            print(f"    [AI Thought]: {thought}")

            # Final Cleanup
            if submission_url and not submission_url.startswith("http"):
                submission_url = urljoin(task_url, submission_url)
            
            # Local Math Fallback (Just in case AI is bad at math)
            # If AI answer matches the CSV sum, we trust it. If it returns 0 but we have a sum, we suggest it.
            if csv_nums and (payload.get("answer") == 0 or payload.get("answer") == "0"):
                 math_sum = sum(csv_nums)
                 if math_sum > 0:
                     print(f"    [Logic] AI returned 0 but CSV has data. Overriding with Sum: {math_sum}")
                     payload["answer"] = int(math_sum) if math_sum.is_integer() else math_sum

            print(f"    Final Answer: {payload.get('answer')}")
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