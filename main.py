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
    """Downloads audio, converts to WAV, and transcribes via Google Speech."""
    print(f"    [Tool] 🔊 Processing Audio: {url}")
    try:
        resp = requests.get(url, timeout=30)
        if 'text/html' in resp.headers.get('Content-Type', ''): return ""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            orig_path = os.path.join(temp_dir, "original")
            with open(orig_path, "wb") as f: f.write(resp.content)
            
            wav_path = os.path.join(temp_dir, "converted.wav")
            AudioSegment.from_file(orig_path).export(wav_path, format="wav")
            
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
                text = r.recognize_google(audio)
            
            print(f"    [Tool] Transcription: {text[:50]}...")
            return f"\n[AUDIO INSTRUCTIONS]: \"{text}\"\n"
    except Exception as e:
        print(f"    [Tool] Audio Error: {e}")
        return ""

async def process_csv_url(url: str) -> list:
    """Downloads CSV and returns a LIST of numbers for Python to calculate."""
    print(f"    [Tool] 📊 Fetching CSV Data: {url}")
    try:
        r = requests.get(url, timeout=15)
        content = r.content.decode("utf-8", errors="replace")
        # Extract all numbers using regex
        nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', content.replace(',', ''))]
        return nums
    except:
        return []

def execute_math_logic(numbers: list, operation: str, threshold: float = None):
    """
    Executes the logic decided by the AI.
    """
    if not numbers: return 0
    
    filtered_nums = numbers
    if threshold is not None:
        if "greater" in operation or ">" in operation:
            filtered_nums = [n for n in numbers if n > threshold]
        elif "less" in operation or "<" in operation:
            filtered_nums = [n for n in numbers if n < threshold]
    
    if "sum" in operation:
        val = sum(filtered_nums)
    elif "count" in operation:
        val = len(filtered_nums)
    elif "max" in operation:
        val = max(filtered_nums)
    elif "min" in operation:
        val = min(filtered_nums)
    elif "average" in operation:
        val = sum(filtered_nums) / len(filtered_nums) if filtered_nums else 0
    else:
        val = sum(numbers) # Default to sum
        
    return int(val) if val.is_integer() else val

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
            
            # Smart Scraping: Get visible text AND specific tag values
            page_text = await page.evaluate("""() => {
                return document.body.innerText + "\\n" + 
                       Array.from(document.querySelectorAll('code, pre, .secret, .code')).map(e => e.innerText).join("\\n");
            }""")

            # 1. EXTRACT LINKS
            media_links = await page.evaluate("""() => {
                let links = [];
                document.querySelectorAll('a').forEach(a => { if (a.href) links.push(a.href) });
                document.querySelectorAll('audio, source').forEach(el => { if (el.src) links.push(el.src) });
                return links;
            }""")

            evidence_text = ""
            csv_numbers = []
            
            # 2. PROCESS FILES
            seen = set()
            for url in media_links:
                if url in seen or url.rstrip('/') == task_url.rstrip('/'): continue
                seen.add(url)
                
                u_low = url.lower()
                # Audio
                if u_low.endswith(('.mp3', '.wav', '.opus', '.m4a')) or 'audio' in u_low and not u_low.endswith('.csv'):
                    evidence_text += await process_audio_google(url)
                # CSV
                elif u_low.endswith('.csv') or ('csv' in u_low and 'download' in u_low):
                    csv_numbers = await process_csv_url(url)
                    evidence_text += f"\n[CSV FILE FOUND]: Contains {len(csv_numbers)} numbers.\n"

            # 3. ASK AI FOR THE PLAN (Not the answer)
            prompt = f"""
            You are a Logic Extraction Engine.
            
            === PAGE TEXT ===
            {page_text}
            
            === AUDIO / EVIDENCE ===
            {evidence_text}
            
            === MISSION ===
            Analyze the instructions. Return a JSON object describing HOW to solve the problem.
            
            1. SCRAPING: Is there a secret code mentioned on the page (e.g. "Cutoff is 5000", "Secret: XYZ")? Extract it.
            2. MATH LOGIC: What calculation should be performed on the CSV numbers? (e.g. "sum numbers greater than 5000").
            
            Return JSON ONLY format:
            {{
                "submission_url": "URL found on page or /submit",
                "secret_code_found": "The code found in text (or null)",
                "math_operation": "sum/count/average/max",
                "math_filter": "greater_than/less_than/none",
                "math_threshold": 12345 (number found in text/audio, or null if none)
            }}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            print(f"    [AI Plan] {plan}")

            # 4. EXECUTE THE PLAN (Python Logic)
            final_answer = None
            
            # Priority A: Scraped Secret (Level 2 Fix)
            if plan.get("secret_code_found"):
                final_answer = plan["secret_code_found"]
            
            # Priority B: Math Calculation (Level 3 Fix)
            elif csv_numbers:
                op = plan.get("math_operation", "sum")
                filt = plan.get("math_filter", "none")
                thresh = plan.get("math_threshold")
                
                # Convert threshold to number if needed
                if thresh is not None:
                    try: thresh = float(str(thresh).replace(',',''))
                    except: thresh = None
                
                # Combine op and filter for the helper
                op_key = f"{op}_{filt}"
                final_answer = execute_math_logic(csv_numbers, op_key, thresh)
                print(f"    [Math] Executed {op} on {len(csv_numbers)} nums with threshold {thresh} -> {final_answer}")

            # Priority C: Fallback to Student Secret
            if not final_answer:
                # If page explicitly asks for "your secret", use it.
                if "secret" in page_text.lower() and "input" in page_text.lower():
                    final_answer = student_secret
                else:
                    final_answer = student_secret # Default fallback

            # 5. SUBMIT
            submission_url = plan.get("submission_url")
            if submission_url and not submission_url.startswith("http"):
                submission_url = urljoin(task_url, submission_url)
            
            # Payload construction
            payload = {
                "email": email,
                "secret": student_secret,
                "url": task_url,
                "answer": final_answer
            }

            print(f"    Final Answer: {final_answer}")
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
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}