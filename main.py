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
        # Security: Ignore if it downloads a webpage instead of audio
        if 'text/html' in response.headers.get('Content-Type', ''): return "" 
        
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, f"audio_{int(time.time())}.mp3")
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with open(path, "rb") as audio_file:
                # Use Whisper to transcribe
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
        # Security: Ignore if it downloads a webpage
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
            # 1. NAVIGATE & WAIT
            await page.goto(task_url)
            await page.wait_for_selector("body", timeout=15000)
            
            # 2. INTELLIGENT SCRAPING (Code-based, not AI-based)
            # We extract links directly from DOM to ensure we don't miss anything.
            page_text = await page.evaluate("document.body.innerText")
            
            # Extract hrefs for CSVs and SRCs for Audio using JavaScript
            media_links = await page.evaluate("""() => {
                let links = [];
                // Get all Anchor tags
                document.querySelectorAll('a').forEach(a => {
                    if (a.href) links.push({url: a.href, type: 'link'});
                });
                // Get all Audio tags
                document.querySelectorAll('audio, source').forEach(el => {
                    if (el.src) links.push({url: el.src, type: 'audio'});
                });
                return links;
            }""")

            # 3. GATHER EVIDENCE (Download Loop)
            evidence_log = ""
            processed_urls = set()
            
            for item in media_links:
                url = item['url']
                # Skip if already processed or if it's the current page itself
                if url in processed_urls or url.rstrip('/') == task_url.rstrip('/'): continue
                
                # Heuristics to decide what to download
                is_audio = item['type'] == 'audio' or url.endswith('.mp3') or url.endswith('.wav')
                is_csv = url.endswith('.csv') or 'csv' in url.lower() or 'download' in url.lower()
                
                if is_audio:
                    evidence_log += await process_audio_url(url)
                    processed_urls.add(url)
                elif is_csv:
                    # Double check it's not a navigation link
                    if not url.endswith('.html') and '/page/' not in url:
                        evidence_log += await process_csv_url(url)
                        processed_urls.add(url)

            # 4. THE BRAIN (LLM SOLVER)
            # We give the LLM a strong persona to fix the Secret/Email issue.
            prompt = f"""
            You are the user with the following credentials:
            - EMAIL: "{email}"
            - SECRET: "{student_secret}"
            
            You are solving a quiz. 
            
            EVIDENCE COLLECTED:
            {evidence_log}
            
            WEBPAGE TEXT:
            '''
            {page_text}
            '''
            
            INSTRUCTIONS:
            1. Parse the page to find the Submission JSON format and URL.
            2. Answer the question based on the EVIDENCE and WEBPAGE TEXT.
            3. IDENTITY RULE: If the question asks for "your secret", "password", or "email", you MUST output your actual credentials (from above), NOT the placeholder text.
            4. DATA RULE: If the question requires calculation, perform it using the data in the EVIDENCE block.
            
            Return JSON ONLY:
            {{
                "submission_url": "...",
                "payload": {{ ...correct json payload... }}
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
            
            # Handle relative URLs
            if submission_url and not submission_url.startswith("http"):
                submission_url = urljoin(task_url, submission_url)

            print(f"    AI Answer: {payload.get('answer')}")
            print(f"    Submitting to: {submission_url}")

            # 5. SUBMIT & RECURSE
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