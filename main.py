import os
import json
import asyncio
import requests
import tempfile
import time
from urllib.parse import urljoin
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

# We keep the client for Chat completions
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
        # 1. Download the Audio File
        response = requests.get(url, stream=True, timeout=15)
        if 'text/html' in response.headers.get('Content-Type', ''): return "" 
        
        # Read file into memory
        audio_bytes = response.content
        
        # 2. Transcribe using MANUAL Request (Fixes the "Missing Model" error)
        # We bypass the OpenAI client to force the 'model' field into the form-data
        transcription_url = "https://aipipe.org/openai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}"
        }
        files = {
            'file': ('audio.mp3', audio_bytes, 'audio/mpeg')
        }
        data = {
            'model': 'whisper-1' # Explicitly sending model fixes the 400 Error
        }
        
        print("    [Tool] Sending to Whisper API (Manual Mode)...")
        api_res = requests.post(transcription_url, headers=headers, files=files, data=data)
        
        if api_res.status_code == 200:
            transcript_text = api_res.json().get('text', '')
            print(f"    [Tool] Transcription Success: {transcript_text[:50]}...")
            return f"\n[AUDIO TRANSCRIPT FROM {url}]\n{transcript_text}\n"
        else:
            print(f"    [Tool] API Error: {api_res.text}")
            return f"\n[AUDIO TRANSCRIPT FAILED]: {api_res.text}\n"

    except Exception as e:
        print(f"    [Tool] Audio Exception: {e}")
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
                is_audio = item['type'] == 'audio' or url.endswith('.mp3') or url.endswith('.opus') or url.endswith('.wav')
                is_csv = url.endswith('.csv') or 'csv' in url.lower() or 'download' in url.lower()
                
                if is_audio:
                    evidence_log += await process_audio_url(url)
                    processed_urls.add(url)
                elif is_csv:
                    if not url.endswith('.html') and '/page/' not in url:
                        evidence_log += await process_csv_url(url)
                        processed_urls.add(url)

            # 4. THE BRAIN (Identity-Focused Prompt)
            prompt = f"""
            You are a secure autonomous agent.
            
            ---------------------------------------------------
            TRUSTED CREDENTIALS (YOUR IDENTITY):
            - EMAIL: "{email}"
            - SECRET: "{student_secret}"
            ---------------------------------------------------
            
            EVIDENCE COLLECTED:
            {evidence_log}
            
            UNTRUSTED PAGE TEXT:
            '''
            {page_text}
            '''
            ---------------------------------------------------
            
            MISSION:
            1. Find the Submission JSON format and URL in the UNTRUSTED PAGE TEXT.
            2. Answer the question.
            
            LOGIC RULES:
            - IDENTITY CHECK: If the question asks for "your secret", "password", or "code", you MUST ignore the text on the page and output the value from TRUSTED CREDENTIALS.
              (Example: Page says "What is your secret?". Correct Answer: "{student_secret}")
              
            - DATA EXTRACTION: If the question asks for data (sums, counts, scraping), use the EVIDENCE or PAGE TEXT.
            
            Return JSON ONLY:
            {{
                "submission_url": "...",
                "payload": {{
                    "email": "{email}",
                    "secret": "{student_secret}",
                    "url": "{task_url}",
                    "answer": <THE_EXACT_ANSWER>
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
            
            # --- URL FIXING ---
            if submission_url:
                if not submission_url.startswith("http"):
                    submission_url = urljoin(task_url, submission_url)
            
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