import os
import json
import asyncio
import requests
import tempfile
import time
import csv
import io
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
# 2. MEDIA HANDLING TOOLS
# ==========================================

def get_full_file_url(base_url: str, relative_url: str) -> str:
    if not relative_url or relative_url == "NONE": return None
    return urljoin(base_url, relative_url)

async def transcribe_audio_file(file_url: str) -> str:
    """Downloads audio and uses OpenAI Whisper for transcription."""
    print(f"    [Tool] Downloading Audio: {file_url}")
    try:
        response = requests.get(file_url, stream=True, timeout=15)
        response.raise_for_status()
        
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
        return f"ERROR: Audio processing failed: {e}"

async def read_csv_file(file_url: str) -> str:
    """Downloads and reads a CSV file into text."""
    print(f"    [Tool] Reading CSV: {file_url}")
    try:
        response = requests.get(file_url, timeout=15)
        response.raise_for_status()
        # Decode and return content
        return response.content.decode('utf-8')
    except Exception as e:
        return f"ERROR: CSV processing failed: {e}"

# ==========================================
# 3. THE SOLVER LOGIC
# ==========================================

async def solve_quiz_task(task_url: str, email: str, student_secret: str, context_log: str = ""):
    """
    context_log: Accumulates data found (Transcriptions, CSVs) so we don't lose it on recursion.
    """
    # CRITICAL FIX: Stop if the URL is None (Prevents crash)
    if not task_url:
        print("    🛑 Stopped: No valid URL provided.")
        return

    print(f"\n[+] Processing Task: {task_url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            # A. Scrape Page
            await page.goto(task_url)
            await page.wait_for_selector("body", timeout=15000)
            content = await page.evaluate("document.body.innerText")
            print(f"    Scraped content length: {len(content)} chars")

            # ==========================================================
            # B. INTELLIGENT AGENT LOOP (Find Missing Data)
            # ==========================================================
            
            # 1. Check for CSV (if mentioned but not yet read)
            # We check if 'csv' is in text AND we haven't already added it to context
            if "csv" in content.lower() and "CSV_CONTENT:" not in context_log:
                print("    [Agent] Detected CSV reference. Extracting URL...")
                link_prompt = f"Find the CSV file URL in this text: '''{content}'''. Return ONLY the URL string."
                csv_link_res = client.chat.completions.create(
                    model="gpt-4o-mini", messages=[{"role": "user", "content": link_prompt}]
                )
                csv_rel_url = csv_link_res.choices[0].message.content.strip()
                full_csv_url = get_full_file_url(task_url, csv_rel_url)
                
                if full_csv_url and "http" in full_csv_url:
                    csv_data = await read_csv_file(full_csv_url)
                    # Add to context and RECURSE (Run again with new knowledge)
                    new_context = context_log + f"\n\n=== CSV_CONTENT ===\n{csv_data}\n"
                    print("    [Agent] CSV Read. Rerunning with new data...")
                    await browser.close()
                    await solve_quiz_task(task_url, email, student_secret, new_context)
                    return

            # 2. Check for Audio (if mentioned but not yet transcribed)
            if ("audio" in content.lower() or ".mp3" in content.lower()) and "AUDIO_TRANSCRIPT:" not in context_log:
                print("    [Agent] Detected Audio reference. Extracting URL...")
                link_prompt = f"Find the Audio/MP3 file URL in this text: '''{content}'''. Return ONLY the URL string."
                audio_link_res = client.chat.completions.create(
                    model="gpt-4o-mini", messages=[{"role": "user", "content": link_prompt}]
                )
                audio_rel_url = audio_link_res.choices[0].message.content.strip()
                full_audio_url = get_full_file_url(task_url, audio_rel_url)
                
                if full_audio_url and "http" in full_audio_url:
                    transcript = await transcribe_audio_file(full_audio_url)
                    # Add to context and RECURSE
                    new_context = context_log + f"\n\n=== AUDIO_TRANSCRIPT ===\n{transcript}\n"
                    print("    [Agent] Audio Transcribed. Rerunning with new data...")
                    await browser.close()
                    await solve_quiz_task(task_url, email, student_secret, new_context)
                    return

            # ==========================================================
            # C. FINAL SOLVER (With all collected context)
            # ==========================================================
            
            full_prompt = f"""
            You are a Data Extractor.
            
            INTERNAL DATA:
            - Email: "{email}"
            - Secret: "{student_secret}"
            
            COLLECTED EVIDENCE (Files/Audio found):
            {context_log}
            
            PAGE TEXT:
            '''
            {content}
            '''
            
            TASK:
            1. Find the Submission URL.
            2. Solve the question.
            
            RULES:
            - If asked for sum/calculation, use the CSV_CONTENT provided above.
            - If asked for a specific value mentioned in audio, use AUDIO_TRANSCRIPT.
            - If asked for identity, use INTERNAL DATA.
            
            OUTPUT JSON ONLY:
            {{
                "submission_url": "...",
                "payload": {{
                    "email": "{email}",
                    "secret": "{student_secret}",
                    "url": "{task_url}",
                    "answer": <THE_ANSWER>
                }}
            }}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": full_prompt}],
                response_format={"type": "json_object"}
            )

            ai_data = json.loads(response.choices[0].message.content)
            submission_url = get_full_file_url(task_url, ai_data.get("submission_url"))
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
                        # CRITICAL: Only recurse if a valid URL exists
                        if result.get("url"):
                            print(f"    Found next level: {result['url']}")
                            await browser.close()
                            await solve_quiz_task(result["url"], email, student_secret)
                            return
                        else:
                            print("    🎉 Quiz Complete!")
                            return
                    else:
                        print("    ❌ Incorrect.")
                        # SAFETY CHECK: Only recurse if a valid NEW url is provided (skipping)
                        if result.get("url"):
                            print("    Retrying next URL provided by server...")
                            await browser.close()
                            await solve_quiz_task(result["url"], email, student_secret)
                            return
                        else:
                            print("    🛑 Game Over. No next URL to proceed.")

                except Exception:
                     print(f"    ❌ Error parsing response: {submit_response.text}")

        except Exception as e:
            print(f"    Error during processing: {e}")
        
        if browser.is_connected():
            await browser.close()

@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}