import os
import json
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

# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
MY_SECRET = os.getenv("MY_SECRET")

client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url="https://aipipe.org/openai/v1"
)
app = FastAPI()

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# ==========================================
# 2. FILE PROCESSING TOOLS
# ==========================================

async def process_audio_url(url: str) -> str:
    print(f"    [Tool] 🔊 Found Audio: {url}")
    try:
        response = requests.get(url, stream=True, timeout=15)
        if 'text/html' in response.headers.get('Content-Type', ''): return "" 
        
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, f"audio_{int(time.time())}.mp3")
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with open(path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            print(f"    [Tool] Transcription: {transcript.text[:50]}...")
            return f"\n[AUDIO TRANSCRIPT FROM {url}]\n{transcript.text}\n"
    except Exception as e:
        print(f"    [Tool] Audio Error: {e}")
        return ""

async def process_csv_url(url: str) -> str:
    print(f"    [Tool] 📊 Found CSV/Data: {url}")
    try:
        response = requests.get(url, timeout=15)
        if 'text/html' in response.headers.get('Content-Type', ''): return ""
        
        content = response.content.decode('utf-8')
        print(f"    [Tool] CSV Content: {len(content)} chars")
        return f"\n[CSV FILE CONTENT FROM {url}]\n{content}\n"
    except Exception as e:
        print(f"    [Tool] CSV Error: {e}")
        return ""

# ==========================================
# 3. MAIN SOLVER LOGIC
# ==========================================

async def solve_quiz_task(task_url: str, email: str, student_secret: str):
    if not task_url: return
    print(f"\n[+] Processing Task: {task_url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            # 1. NAVIGATE
            await page.goto(task_url)
            await page.wait_for_selector("body", timeout=15000)
            
            # 2. EXTRACT CONTENT
            page_text = await page.evaluate("document.body.innerText")
            
            # 3. CHECK FOR FILES (Code-based detection)
            # We look for links to CSVs or Audio sources
            media_links = await page.evaluate("""() => {
                let links = [];
                document.querySelectorAll('a').forEach(a => {
                    if (a.href) links.push({url: a.href, type: 'link'});
                });
                document.querySelectorAll('audio, source').forEach(el => {
                    if (el.src) links.push({url: el.src, type: 'audio'});
                });
                return links;
            }""")

            evidence_log = ""
            processed_urls = set()
            
            for item in media_links:
                url = item['url']
                if url in processed_urls or url.rstrip('/') == task_url.rstrip('/'): continue
                
                # Filter for relevant files
                is_audio = item['type'] == 'audio' or url.endswith('.mp3') or url.endswith('.wav')
                is_csv = url.endswith('.csv') or 'csv' in url.lower() or 'download' in url.lower()
                
                if is_audio:
                    evidence_log += await process_audio_url(url)
                    processed_urls.add(url)
                elif is_csv:
                    if not url.endswith('.html') and '/page/' not in url:
                        evidence_log += await process_csv_url(url)
                        processed_urls.add(url)

            # 4. THE BRAIN (Revised Prompt)
            prompt = f"""
            You are a precise data extraction agent.
            
            USER IDENTITY (Use ONLY if asked for "your" credentials):
            - EMAIL: "{email}"
            - SECRET: "{student_secret}"
            
            TASK:
            1. Analyze the PAGE TEXT and EVIDENCE below.
            2. Extract the Submission URL (it might be a relative path like /submit).
            3. Answer the question.
            
            EVIDENCE (Audio/Files):
            {evidence_log}
            
            PAGE TEXT:
            '''
            {page_text}
            '''
            
            RULES FOR ANSWERING:
            - IDENTITY: If asked for "your secret", "password", or "email", output the USER IDENTITY values.
            - EXTRACTION: If asked to "scrape" or "find" a code on the page, look closely at the PAGE TEXT. It might be a random string like "H5K9". Extract it exactly.
            - CALCULATION: If asked for a sum, use the EVIDENCE data.
            
            Return JSON ONLY:
            {{
                "submission_url": "...",
                "payload": {{
                    "email": "{email}",
                    "secret": "{student_secret}",
                    "url": "{task_url}",
                    "answer": <THE_EXACT_ANSWER_VALUE>
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
            
            # --- ROBUST URL FIXING ---
            if submission_url:
                if not submission_url.startswith("http"):
                    submission_url = urljoin(task_url, submission_url)
            
            # Ensure the payload URL is also valid
            if "url" in payload and not payload["url"].startswith("http"):
                 payload["url"] = task_url

            print(f"    AI Answer: {payload.get('answer')}")
            print(f"    Submitting to: {submission_url}")

            # 5. SUBMIT
            if submission_url:
                submit_res = requests.post(submission_url, json=payload)
                try:
                    res_json = submit_res.json()
                    print(f"    Server: {res_json}")
                    
                    if res_json.get("correct") == True:
                        print("    ✅ Correct!")
                        if res_json.get("url"):
                            await browser.close()
                            await solve_quiz_task(res_json["url"], email, student_secret)
                    elif res_json.get("url"):
                         print("    ⚠️ Incorrect, but retrying next level...")
                         await browser.close()
                         await solve_quiz_task(res_json["url"], email, student_secret)
                    else:
                        print("    ❌ Game Over.")
                        
                except Exception:
                    print(f"    ❌ Error: {submit_res.text}")

        except Exception as e:
            print(f"    Error: {e}")
        
        if browser.is_connected():
            await browser.close()

@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}