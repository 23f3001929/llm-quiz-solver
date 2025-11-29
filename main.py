import os
import json
import re
import asyncio
import requests
import tempfile
import csv
import io
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

async def process_csv_url(url: str) -> dict:
    """Downloads CSV and returns content, parsed numbers, and raw lines."""
    print(f"    [Tool] 📊 Fetching CSV Data: {url}")
    try:
        r = requests.get(url, timeout=15)
        content = r.content.decode("utf-8", errors="replace")
        
        # Extract all numbers for math
        nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', content.replace(',', ''))]
        
        # Parse properly to handle structure if needed
        rows = []
        try:
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
        except:
            pass

        return {
            "content": content, 
            "numbers": nums, 
            "lines": content.splitlines(),
            "rows": rows
        }
    except:
        return {"content": "", "numbers": [], "lines": [], "rows": []}

def execute_logic(csv_data: dict, operation: str, threshold: float = None, email: str = "", mod_n: int = 0):
    """Executes math OR list filtering."""
    numbers = csv_data.get("numbers", [])
    lines = csv_data.get("lines", [])
    
    # --- LIST FILTERING MODE (For "Return a JSON array") ---
    # Used when server asks for "rows" or "items" instead of a sum
    if "filter" in operation and "list" in operation:
        results = []
        # Fallback if no threshold found but filter requested: return all
        if threshold is None: 
            return lines[:50] # Limit to avoid huge payload if logic fails
            
        for line in lines:
            # Find numbers in this line
            line_nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', line.replace(',', ''))]
            if not line_nums: continue
            
            # Check condition
            match = False
            if "greater" in operation or ">" in operation:
                match = any(n > threshold for n in line_nums)
            elif "less" in operation or "<" in operation:
                match = any(n < threshold for n in line_nums)
            
            if match:
                results.append(line.strip())
        
        print(f"    [Logic] Filtered list size: {len(results)}")
        return results

    # --- MATH MODE ---
    if not numbers: return 0
    
    filtered_nums = numbers
    if threshold is not None:
        if "greater" in operation or ">" in operation:
            filtered_nums = [n for n in numbers if n > threshold]
        elif "less" in operation or "<" in operation:
            filtered_nums = [n for n in numbers if n < threshold]
    
    # Default to sum if list is empty to prevent crash
    if not filtered_nums: filtered_nums = numbers 

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
        
    # LOGS CHALLENGE: "Sum bytes + email length mod N"
    # We apply the modulo offset if the AI detected it or if passed explicitly
    if mod_n > 0:
        offset = len(email) % mod_n
        print(f"    [Math] Adding dynamic offset (email len {len(email)} % {mod_n}) = {offset}")
        val += offset
        
    # INVOICE CHALLENGE: Round to 2 decimals
    if "decimal" in operation or "invoice" in operation:
        return round(val, 2)
        
    # FIX: Safely convert float to int if strictly integer
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val

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

            # 1. EXTRACT LINKS & IMAGES
            media = await page.evaluate("""() => {
                let items = [];
                document.querySelectorAll('a').forEach(a => { if (a.href) items.push({url: a.href, type: 'link'}) });
                document.querySelectorAll('audio, source').forEach(el => { if (el.src) items.push({url: el.src, type: 'audio'}) });
                document.querySelectorAll('img').forEach(el => { if (el.src) items.push({url: el.src, type: 'image'}) });
                return items;
            }""")

            evidence_text = ""
            csv_data = {"content": "", "numbers": [], "lines": [], "rows": []}
            image_urls = []
            
            # 2. PROCESS MEDIA
            seen = set()
            for item in media:
                url = item['url']
                if url in seen or url.rstrip('/') == task_url.rstrip('/'): continue
                seen.add(url)
                
                u_low = url.lower()
                
                if item['type'] == 'image':
                    image_urls.append(url)
                    
                elif u_low.endswith(('.mp3', '.wav', '.opus', '.m4a')) or item['type'] == 'audio':
                    if not u_low.endswith('.csv'):
                        evidence_text += await process_audio_google(url)
                        
                elif u_low.endswith('.csv') or ('csv' in u_low and 'download' in u_low):
                    csv_data = await process_csv_url(url)
                    evidence_text += f"\n[CSV FILE FOUND]: Contains {len(csv_data['numbers'])} numbers.\n"

            # 3. ANALYZE PAGE FOR MATH RULES
            # Regex to find "mod N" or "modulo N"
            mod_match = re.search(r'mod(?:ulo)?\s*(\d+)', page_text.lower() + evidence_text.lower())
            mod_n = int(mod_match.group(1)) if mod_match else 0

            # 4. ASK AI FOR THE PLAN
            messages = [
                {"role": "system", "content": "You are a Logic Extraction Engine."}
            ]
            
            user_text_prompt = f"""
            === USER IDENTITY ===
            EMAIL: "{email}"
            SECRET: "{student_secret}"
            
            === PAGE TEXT ===
            {page_text}
            
            === AUDIO / EVIDENCE ===
            {evidence_text}
            
            === MISSION ===
            1. COMMANDS: "Craft a command" (uv/git/curl) -> Output command string.
            2. LISTS: "Return a JSON array", "Filter rows" -> Use 'filter_list'.
            3. MATH: "Sum", "Count", "Total" -> Use 'sum'/'count'.
            4. GENERAL: "What is...", "Code?", "Color?" -> Extract exact value. Avoid placeholders like "#rrggbb".
            """
            
            user_content = [{"type": "text", "text": user_text_prompt}]
            
            # Add images for Vision
            for img_url in image_urls:
                user_content.append({"type": "image_url", "image_url": {"url": img_url}})
            
            user_content.append({"type": "text", "text": """
                Return JSON ONLY:
                {
                    "submission_url": "URL found on page",
                    "generated_command": "Command string (or null)",
                    "general_answer": "Extracted text/color/link (or null)",
                    "math_operation": "sum/count/average/max/filter_list (or null)",
                    "math_filter": "greater_than/less_than/none",
                    "math_threshold": 12345 (number or null),
                    "is_invoice_or_decimal": boolean
                }
            """})
            
            messages.append({"role": "user", "content": user_content})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            print(f"    [AI Plan] {plan}")

            # 5. EXECUTE THE PLAN
            final_answer = None
            
            gen_cmd = plan.get("generated_command")
            if gen_cmd and ("curl" in gen_cmd or "uv" in gen_cmd or "git" in gen_cmd or "grep" in gen_cmd):
                final_answer = gen_cmd
            
            elif plan.get("general_answer") and str(plan["general_answer"]).lower() not in ["hello", "your secret", "email", "code", "null", "#rrggbb"]:
                final_answer = plan["general_answer"]
            
            # Fallback Logic
            if not final_answer and csv_data['numbers']:
                op = plan.get("math_operation", "sum") or "sum"
                filt = plan.get("math_filter", "none")
                thresh = plan.get("math_threshold")
                
                if thresh is not None:
                    try: thresh = float(str(thresh).replace(',',''))
                    except: thresh = None
                
                op_key = f"{op}_{filt}"
                if plan.get("is_invoice_or_decimal"): op_key += "_decimal"
                
                final_answer = execute_logic(csv_data, op_key, thresh, email, mod_n)
                print(f"    [Math] Executed {op} -> {final_answer}")

            # Ultimate Fallback
            if not final_answer:
                if "secret" in page_text.lower() and "input" in page_text.lower():
                    final_answer = student_secret
                else:
                    final_answer = student_secret

            # 6. SUBMIT (Safety Checks)
            submission_url = plan.get("submission_url")
            if submission_url:
                if not submission_url.startswith("http"):
                    submission_url = urljoin(task_url, submission_url)
            
            # Prevent infinite loop by redirecting self-submission to /submit
            parsed_task = urlparse(task_url)
            parsed_sub = urlparse(submission_url) if submission_url else parsed_task
            
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
                    print(f"    ❌ Server returned non-JSON. Status: {res.status_code}")
                    print(f"    Response Body: {res.text[:300]}")

        except Exception as e:
            print("    Error:", e)
        
        if browser.is_connected():
            await browser.close()

@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}