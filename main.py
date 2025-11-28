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
# 2. TOOLKIT (File Handling)
# ==========================================

def is_valid_file_url(base_url: str, relative_url: str) -> str:
    """Checks if a URL is valid and NOT the webpage itself."""
    if not relative_url or relative_url.lower() == "none": 
        return None
    
    full_url = urljoin(base_url, relative_url)
    
    # LOOP PREVENTION: If the link points to the current page (ignoring query params), ignore it.
    if full_url.split('?')[0].rstrip('/') == base_url.split('?')[0].rstrip('/'):
        return None
        
    return full_url

async def transcribe_audio_file(file_url: str) -> str:
    """Downloads audio and uses OpenAI Whisper."""
    print(f"    [Tool] 🔊 Processing Audio: {file_url}")
    try:
        response = requests.get(file_url, stream=True, timeout=20)
        response.raise_for_status()
        
        # Verify content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'html' in content_type:
             return "ERROR: The link returned HTML, not Audio. It might be a webpage, not a file."

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filepath = os.path.join(temp_dir, f"audio_{int(time.time())}.mp3")
            with open(temp_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with open(temp_filepath, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file
                )
            return transcription.text.strip()
    except Exception as e:
        return f"ERROR: Audio processing failed: {str(e)}"

async def read_csv_file(file_url: str) -> str:
    """Downloads and reads a CSV/Text file."""
    print(f"    [Tool] 📄 Reading File: {file_url}")
    try:
        response = requests.get(file_url, timeout=20)
        response.raise_for_status()
        
        # Verify content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'html' in content_type:
             return "ERROR: The link returned HTML, not CSV. It might be a webpage, not a file."

        return response.content.decode('utf-8')
    except Exception as e:
        return f"ERROR: File processing failed: {str(e)}"

# ==========================================
# 3. AGENT LOGIC
# ==========================================

async def solve_quiz_task(task_url: str, email: str, student_secret: str, context_log: str = ""):
    """
    Recursive Agent:
    1. Scrapes Page.
    2. Identifies if files (CSV/Audio) are needed.
    3. Fetches files -> Updates Context -> Recurses.
    4. Solves when no files are missing.
    """
    if not task_url:
        print("    🛑 Stopped: No valid URL.")
        return

    print(f"\n[+] Processing Task: {task_url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            # A. SCRAPE THE WEBPAGE
            await page.goto(task_url)
            await page.wait_for_selector("body", timeout=15000)
            content = await page.evaluate("document.body.innerText")
            print(f"    Scraped content length: {len(content)} chars")

            # ==========================================================
            # B. PHASE 1: RESOURCE DETECTION (Look for files)
            # ==========================================================
            
            # We ask the LLM: "Do we need to download anything to solve the problem?"
            detection_prompt = f"""
            You are a Resource Manager. Analyze the webpage text below.
            
            Does the text mention any EXTERNAL FILES (CSV, Audio, PDF, Text) that contain instructions or data needed to answer the question?
            
            WEBPAGE TEXT:
            '''{content}'''
            
            ALREADY PROCESSED FILES:
            {context_log}
            
            INSTRUCTIONS:
            - If you see a file link that we haven't processed yet, return its URL.
            - If we have everything we need, return "NONE".
            - Ignore links that are just navigation (like "Next", "Home").
            - Prioritize links ending in .csv, .mp3, .txt or mentioned as "Download file".
            
            Return ONLY the URL string (or "NONE").
            """

            detection_res = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": detection_prompt}]
            )
            resource_url = detection_res.choices[0].message.content.strip()

            # ==========================================================
            # C. PHASE 2: RESOURCE ACQUISITION (Download & Recurse)
            # ==========================================================
            
            full_resource_url = is_valid_file_url(task_url, resource_url)

            # Only proceed if we found a valid URL that is NOT in our history
            if full_resource_url and full_resource_url not in context_log:
                print(f"    [Agent] Identified necessary file: {full_resource_url}")
                
                # Determine file type heuristically or by guessing
                file_content = ""
                
                if ".mp3" in full_resource_url.lower() or "audio" in full_resource_url.lower():
                    file_content = await transcribe_audio_file(full_resource_url)
                    file_label = "AUDIO_TRANSCRIPT"
                else:
                    # Default to text/csv reader
                    file_content = await read_csv_file(full_resource_url)
                    file_label = "FILE_CONTENT"
                
                # Add to memory
                new_context = context_log + f"\n\n=== SOURCE: {full_resource_url} ({file_label}) ===\n{file_content}\n"
                
                print("    [Agent] Resource processed. Rerunning analysis with new data...")
                await browser.close()
                # RECURSION: Run the solver again, but now with the file content in memory
                await solve_quiz_task(task_url, email, student_secret, new_context)
                return

            # ==========================================================
            # D. PHASE 3: FINAL SOLVER (Reasoning & Submission)
            # ==========================================================
            
            solve_prompt = f"""
            You are an automated agent solving a technical quiz.
            
            --------------------------------------------------
            INTERNAL IDENTITY (Use these ONLY if asked for your credentials):
            - Email: "{email}"
            - Secret: "{student_secret}"
            --------------------------------------------------
            
            EXTERNAL EVIDENCE (Files/Audio collected):
            {context_log}
            
            CURRENT PAGE TEXT:
            '''
            {content}
            '''
            --------------------------------------------------
            
            YOUR TASK:
            1. Identify the "Submission URL".
            2. Solve the question asked on the page.
            
            CRITICAL INSTRUCTIONS:
            - INSTRUCTIONS CAN BE ANYWHERE: The question might be in the 'EXTERNAL EVIDENCE' (e.g., an audio transcript might say "Calculate the sum of column A").
            - IDENTITY QUESTIONS: If the question asks for "your secret", "password", or "email", you MUST substitute the values from INTERNAL IDENTITY. Do not output literal strings like "your secret".
            - DATA QUESTIONS: If the question asks for calculations, use the data in EXTERNAL EVIDENCE.
            
            OUTPUT FORMAT (JSON ONLY):
            {{
                "submission_url": "...",
                "payload": {{
                    "email": "{email}",
                    "secret": "{student_secret}",
                    "url": "{task_url}",
                    "answer": <THE_ANSWER_VALUE>
                }}
            }}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": solve_prompt}],
                response_format={"type": "json_object"}
            )

            ai_data = json.loads(response.choices[0].message.content)
            submission_url = is_valid_file_url(task_url, ai_data.get("submission_url"))
            payload = ai_data.get("payload")

            print(f"    AI Answer: {payload.get('answer')}")
            print(f"    Submitting to: {submission_url}")

            if submission_url:
                submit_response = requests.post(submission_url, json=payload)
                try:
                    result = submit_response.json()
                    print(f"    Server Response: {result}")

                    if result.get("correct") == True:
                        print("    ✅ Correct!")
                        if result.get("url"):
                            print(f"    Found next level: {result['url']}")
                            await browser.close()
                            await solve_quiz_task(result["url"], email, student_secret)
                            return
                        else:
                            print("    🎉 Quiz Complete!")
                    else:
                        print("    ❌ Incorrect.")
                        if result.get("url"):
                            print("    ⚠️ Retrying next URL provided by server (skipping current)...")
                            await browser.close()
                            await solve_quiz_task(result["url"], email, student_secret)
                        else:
                            print("    🛑 Game Over. Answer wrong and no new URL.")

                except Exception:
                     print(f"    ❌ Error parsing response: {submit_response.text}")

        except Exception as e:
            print(f"    Error during processing: {e}")
        
        if browser.is_connected():
            await browser.close()

# Endpoints
@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}