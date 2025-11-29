import os
import json
import re
import asyncio
import requests
import tempfile
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
        nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', content.replace(',', ''))]
        return nums
    except:
        return []

def execute_math_logic(numbers: list, operation: str, threshold: float = None, email: str = ""):
    """Executes the logic decided by the AI."""
    if not numbers: return 0
    
    filtered_nums = numbers
    if threshold is not None:
        if "greater" in operation or ">" in operation:
            filtered_nums = [n for n in numbers if n > threshold]
        elif "less" in operation or "<" in operation:
            filtered_nums = [n for n in numbers if n < threshold]
    
    if not filtered_nums: return 0 

    val = 0
    if "sum" in operation:
        val = sum(filtered_nums)
    elif "count" in operation:
        val = len(filtered_nums)
    elif "max" in operation:
        val = max(filtered_nums)
    elif "min" in operation:
        val = min(filtered_nums)
    elif "average" in operation:
        val = sum(filtered_nums) / len(filtered_nums)
    else:
        val = sum(numbers)
        
    # LOGS CHALLENGE: "Sum bytes + email length mod 5"
    if "mod 5" in operation or "offset" in operation:
        offset = len(email) % 5
        print(f"    [Math] Adding offset (email len {len(email)} % 5) = {offset}")
        val += offset
        
    # INVOICE CHALLENGE: Round to 2 decimals
    if "decimal" in operation or "invoice" in operation:
        return round(val, 2)
        
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
            
            # Scrape visible text AND special tags
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

            # 3. ASK AI FOR THE PLAN
            prompt = f"""
            You are a Logic Extraction Engine.
            
            === USER IDENTITY ===
            EMAIL: "{email}"
            SECRET: "{student_secret}"
            
            === PAGE TEXT ===
            {page_text}
            
            === AUDIO / EVIDENCE ===
            {evidence_text}
            
            === MISSION ===
            Analyze the instructions and decide the answer strategy.
            
            1. COMMAND CRAFTING: If asked to "Craft a command", construct it exactly.
            2. GENERAL QUESTION: If asked for a specific value (e.g., "What is the link?", "Dominant color?", "Code phrase?"), extract that EXACT string/value.
            3. MATH LOGIC: If calculation needed on CSV, describe it. (e.g. "sum with mod 5 offset").
            
            Return JSON ONLY:
            {{
                "submission_url": "URL found on page",
                "generated_command": "Command string (or null)",
                "general_answer": "Extracted text answer (or null)",
                "math_operation": "sum/count/average/max (or null)",
                "math_filter": "greater_than/less_than/none",
                "math_threshold": 12345 (number or null),
                "is_invoice_or_decimal": boolean
            }}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            print(f"    [AI Plan] {plan}")

            # 4. EXECUTE THE PLAN
            final_answer = None
            
            # Priority A: Command Generation
            if plan.get("generated_command"):
                final_answer = plan["generated_command"]
            
            # Priority B: General Answer (Links, Colors, Phrases)
            elif plan.get("general_answer") and str(plan["general_answer"]).lower() not in ["hello", "your secret", "email", "code", "null"]:
                final_answer = plan["general_answer"]
            
            # Priority C: Math Calculation
            elif csv_numbers:
                op = plan.get("math_operation", "sum") or "sum"
                filt = plan.get("math_filter", "none")
                thresh = plan.get("math_threshold")
                
                if thresh is not None:
                    try: thresh = float(str(thresh).replace(',',''))
                    except: thresh = None
                
                # Check for special modifiers in operation name
                op_key = f"{op}_{filt}"
                if plan.get("is_invoice_or_decimal"): op_key += "_decimal"
                # Pass email for "mod 5" logic
                if "mod" in str(page_text).lower() or "offset" in str(page_text).lower(): op_key += "_mod5"

                final_answer = execute_math_logic(csv_numbers, op_key, thresh, email)
                print(f"    [Math] Executed {op} -> {final_answer}")

            # Priority D: Fallback
            if not final_answer:
                if "secret" in page_text.lower() and "input" in page_text.lower():
                    final_answer = student_secret
                else:
                    final_answer = student_secret 

            # 5. SUBMIT (With Safety Check)
            submission_url = plan.get("submission_url")
            
            if submission_url:
                if not submission_url.startswith("http"):
                    submission_url = urljoin(task_url, submission_url)
            
            parsed_task = urlparse(task_url)
            parsed_sub = urlparse(submission_url)
            
            if parsed_sub.path == parsed_task.path:
                print("    ⚠️ AI tried to submit to the Task URL. Redirecting to /submit...")
                submission_url = f"{parsed_task.scheme}://{parsed_task.netloc}/submit"

            payload = {
                "email": email,
                "secret": student_secret,
                "url": task_url,
                "answer": final_answer
            }

            print(f"    Final Answer: {final_answer}")
            print(f"    Submitting to: {submission_url}")

            if submission_url:
                res = requests.post(submission_url, json=payload)
                
                try:
                    res_json = res.json()
                    print("    Server:", res_json)
                    if res_json.get("url"):
                        await browser.close()
                        await solve_quiz_task(res_json["url"], email, student_secret)
                except Exception:
                    print(f"    ❌ Server returned non-JSON error. Status: {res.status_code}")
                    print(f"    Response Body: {res.text[:300]}")

        except Exception as e:
            print("    Error:", e)
        
        if browser.is_connected():
            await browser.close()

@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}