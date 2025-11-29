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

async def process_csv_url(url: str) -> dict:
    """Downloads CSV and returns both raw text and numbers."""
    print(f"    [Tool] 📊 Fetching CSV Data: {url}")
    try:
        r = requests.get(url, timeout=15)
        content = r.content.decode("utf-8", errors="replace")
        # Extract numbers for math
        nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', content.replace(',', ''))]
        # Return object with content for filtering
        return {"content": content, "numbers": nums, "lines": content.splitlines()}
    except:
        return {"content": "", "numbers": [], "lines": []}

def execute_logic(csv_data: dict, operation: str, threshold: float = None, email: str = ""):
    """Executes math OR list filtering."""
    numbers = csv_data.get("numbers", [])
    lines = csv_data.get("lines", [])
    
    # --- LIST FILTERING MODE (For "Return a JSON array") ---
    if "filter_list" in operation:
        # Simple heuristic: return lines/rows containing numbers > threshold
        results = []
        if threshold is not None:
            for line in lines:
                # Find numbers in this line
                line_nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', line.replace(',', ''))]
                if any(n > threshold for n in line_nums) if "greater" in operation else any(n < threshold for n in line_nums):
                    results.append(line.strip())
        return results # Return list, not number

    # --- MATH MODE ---
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
        
    # Special Logics
    if "mod 5" in operation or "offset" in operation:
        offset = len(email) % 5
        val += offset
        
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

            # 1. EXTRACT LINKS & IMAGES
            media = await page.evaluate("""() => {
                let items = [];
                document.querySelectorAll('a').forEach(a => { if (a.href) items.push({url: a.href, type: 'link'}) });
                document.querySelectorAll('audio, source').forEach(el => { if (el.src) items.push({url: el.src, type: 'audio'}) });
                document.querySelectorAll('img').forEach(el => { if (el.src) items.push({url: el.src, type: 'image'}) });
                return items;
            }""")

            evidence_text = ""
            csv_data = {"content": "", "numbers": [], "lines": []}
            image_urls = []
            
            # 2. PROCESS MEDIA
            seen = set()
            for item in media:
                url = item['url']
                if url in seen or url.rstrip('/') == task_url.rstrip('/'): continue
                seen.add(url)
                
                u_low = url.lower()
                
                if item['type'] == 'image':
                    image_urls.append(url) # Save for Vision
                    
                elif u_low.endswith(('.mp3', '.wav', '.opus', '.m4a')) or item['type'] == 'audio':
                    if not u_low.endswith('.csv'):
                        evidence_text += await process_audio_google(url)
                        
                elif u_low.endswith('.csv') or ('csv' in u_low and 'download' in u_low):
                    csv_data = await process_csv_url(url)
                    evidence_text += f"\n[CSV FILE FOUND]: Contains {len(csv_data['numbers'])} numbers.\n"

            # 3. ASK AI FOR THE PLAN (Multi-Modal)
            
            # Construct Message with Images if available
            messages = [
                {"role": "system", "content": "You are a Logic Extraction Engine. Analyze instructions and evidence."}
            ]
            
            user_content = [
                {"type": "text", "text": f"""
                === USER IDENTITY ===
                EMAIL: "{email}"
                SECRET: "{student_secret}"
                
                === PAGE TEXT ===
                {page_text}
                
                === AUDIO / EVIDENCE ===
                {evidence_text}
                
                === MISSION ===
                1. COMMANDS: If asked to "Craft a command" (uv/git/curl), output the command string.
                   - NOTE: If the task is just "Calculate total and POST it", that is MATH, NOT a command.
                2. MATH/DATA: If asked to sum/count/filter CSV, describe the logic.
                   - NOTE: If asked for a "JSON array" of items, use "filter_list".
                3. GENERAL: If asked for a color (from image), link, or code, extract it.
                """}
            ]
            
            # Add images to prompt for Vision tasks (Heatmap)
            for img_url in image_urls:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })
            
            user_content.append({"type": "text", "text": """
                Return JSON ONLY:
                {
                    "submission_url": "URL found on page",
                    "generated_command": "Command string ONLY if specifically asked to CRAFT a command (otherwise null)",
                    "general_answer": "Extracted text/color/link (or null)",
                    "math_operation": "sum/count/average/max/filter_list (or null)",
                    "math_filter": "greater_than/less_than/none",
                    "math_threshold": 12345 (number or null),
                    "is_invoice_or_decimal": boolean,
                    "use_mod_5": boolean
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

            # 4. EXECUTE THE PLAN
            final_answer = None
            
            # Priority A: CLI Command Generation (Strict check to avoid Invoice false positives)
            if plan.get("generated_command") and "curl" in plan["generated_command"] or "uv" in plan["generated_command"] or "git" in plan["generated_command"]:
                final_answer = plan["generated_command"]
            
            # Priority B: General Answer (Links, Colors, Phrases)
            elif plan.get("general_answer") and str(plan["general_answer"]).lower() not in ["hello", "your secret", "email", "code", "null"]:
                # Fix relative paths for MD task
                ans = plan["general_answer"]
                if ans.endswith(".md") and ans.startswith("/"):
                     # Sometimes server wants full url, sometimes relative. Usually relative is safer if extracted text.
                     pass 
                final_answer = ans
            
            # Priority C: Math / Data Logic
            elif csv_data['numbers']:
                op = plan.get("math_operation", "sum") or "sum"
                filt = plan.get("math_filter", "none")
                thresh = plan.get("math_threshold")
                
                if thresh is not None:
                    try: thresh = float(str(thresh).replace(',',''))
                    except: thresh = None
                
                # Build operation key
                op_key = f"{op}_{filt}"
                if plan.get("is_invoice_or_decimal"): op_key += "_decimal"
                if plan.get("use_mod_5") or "mod" in page_text.lower(): op_key += "_mod5" # Force check text for mod

                final_answer = execute_logic(csv_data, op_key, thresh, email)
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