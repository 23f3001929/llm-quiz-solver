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

# Initialize OpenAI Client
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
    """Combines base URL and relative URL."""
    return urljoin(base_url, relative_url)

async def transcribe_audio_file(file_url: str, task_url: str) -> str:
    """Downloads audio and uses OpenAI Whisper for transcription."""
    print(f"    [Tool] Attempting to download and transcribe audio from: {file_url}")
    
    # 1. Download file content
    try:
        # Use simple requests for file downloads
        response = requests.get(file_url, stream=True, timeout=15)
        response.raise_for_status()
    except Exception as e:
        return f"ERROR: Could not download audio file: {e}"

    # 2. Save file temporarily
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"audio_{int(time.time())}.mp3"
        temp_filepath = os.path.join(temp_dir, file_name)
        
        with open(temp_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # 3. Transcribe using Whisper
        try:
            with open(temp_filepath, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            transcribed_text = transcription.text.strip()
            print(f"    [Tool] Transcription successful. Text length: {len(transcribed_text)}")
            return transcribed_text

        except Exception as e:
            return f"ERROR: OpenAI transcription failed: {e}"


# ==========================================
# 3. THE SOLVER LOGIC
# ==========================================

async def solve_quiz_task(task_url: str, email: str, student_secret: str, additional_context: str = None):
    
    print(f"\n[+] Processing Task: {task_url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            # A. Visit the URL and scrape content
            await page.goto(task_url)
            await page.wait_for_selector("body", timeout=15000)
            content = await page.evaluate("document.body.innerText")
            print(f"    Scraped content length: {len(content)} chars")

            # B. AGENT LOOP: Check if audio needs to be processed
            if additional_context is None and ("audio" in content.lower() or ".mp3" in content.lower()):
                # Step 1: LLM extracts audio URL
                audio_url_prompt = f"""
                Analyze the following webpage content. Extract the URL of any downloadable audio file or the link mentioned next to 'Download file'.
                WEBPAGE CONTENT: '''{content}'''
                Return ONLY the URL as a plain string. If no clear URL is found, return 'NONE'.
                """
                
                audio_url_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": audio_url_prompt}]
                )
                audio_url = audio_url_response.choices[0].message.content.strip()
                
                if audio_url != 'NONE':
                    full_audio_url = get_full_file_url(task_url, audio_url)
                    transcription = await transcribe_audio_file(full_audio_url, task_url)

                    # Step 2: Recurse/Rerun with new context
                    if transcription.startswith("ERROR"):
                         context_payload = f"Media Error: {transcription}"
                    else:
                         context_payload = f"Media Transcription: {transcription}"
                         
                    print(f"    [Agent] Rerunning analysis with transcription context.")
                    await browser.close()
                    # Rerun the solver with the new context
                    await solve_quiz_task(task_url, email, student_secret, context_payload)
                    return

            # C. Build the Context Block (Clean Syntax Fix)
            context_block = ""
            if additional_context:
                context_block = f"""
--- NEW CONTEXT (From Audio/Media) ---
{additional_context}
"""
            
            # D. Universal Solver Prompt (Final Decision)
            full_prompt = f"""
            You are an automated data extraction assistant.
            
            --------------------------------------------------
            INTERNAL CONTEXT (Do not use as answer unless needed):
            - My Email: "{email}"
            - My Secret Code: "{student_secret}"
            --------------------------------------------------
            
            {context_block}
            
            WEBPAGE CONTENT:
            '''
            {content}
            '''
            --------------------------------------------------
            
            YOUR MISSION:
            1. Find the "Submission URL".
            2. Determine the "Correct Answer" to the question based on all provided context.
            
            LOGIC FOR ANSWERING:
            - If the question asks for user identity (email, secret, code, password, etc.), use the INTERNAL CONTEXT.
            - If the question asks for calculation (sum, count) or extraction, use the WEBPAGE CONTENT or the NEW CONTEXT.
            
            OUTPUT FORMAT:
            Return ONLY a valid JSON object. No markdown, no explanations.
            {{
                "submission_url": "...",
                "payload": {{
                    "email": "{email}",
                    "secret": "{student_secret}",
                    "url": "{task_url}",
                    "answer": <THE_CALCULATED_ANSWER>
                }}
            }}
            """

            # E. Get Final Decision
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": full_prompt}],
                response_format={"type": "json_object"}
            )

            # F. Parse and Submit
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

                    # If Correct -> Recurse
                    if result.get("correct") == True and "url" in result:
                        print("    ✅ Answer Correct! Moving to next level...")
                        await browser.close()
                        await solve_quiz_task(result["url"], email, student_secret)
                        return
                    
                    # If Incorrect but a URL is offered (Skip/Next Logic)
                    elif "url" in result:
                         print("    ⚠️ Answer rejected, but proceeding to next URL...")
                         await browser.close()
                         await solve_quiz_task(result["url"], email, student_secret)
                         return

                except Exception:
                     print(f"    ❌ Error parsing response: {submit_response.text}")

        except Exception as e:
            print(f"    Error during processing: {e}")
        
        if browser.is_connected():
            await browser.close()

# Endpoints remain the same
@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    # Start the solver with no initial additional context
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}