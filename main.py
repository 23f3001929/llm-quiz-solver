import os
import json
import asyncio
import requests
from urllib.parse import urljoin
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from openai import OpenAI

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

load_dotenv()

# Get secrets
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
MY_SECRET = os.getenv("MY_SECRET")

# Safety check
if not AIPIPE_TOKEN or not MY_SECRET:
    # Fallback for local testing if .env is missing, but Render needs the env vars set in dashboard
    print("Warning: Secrets might be missing. Ensure AIPIPE_TOKEN and MY_SECRET are set.")

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
# 2. THE SOLVER LOGIC
# ==========================================

async def solve_quiz_task(task_url: str, email: str, student_secret: str):
    print(f"\n[+] Processing Task: {task_url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            # A. Visit the URL
            await page.goto(task_url)
            await page.wait_for_selector("body", timeout=15000)
            
            # B. Scrape the text
            content = await page.evaluate("document.body.innerText")
            print(f"    Scraped content length: {len(content)} chars")

            # C. AI Analysis
            prompt = f"""
            You are a data extraction agent. 
            Analyze the following text from a quiz page:
            ----------------
            {content}
            ----------------
            
            Context Data:
            - User Email: "{email}"
            - User Secret: "{student_secret}"
            - Current Page URL: "{task_url}"
            
            Instructions:
            1. Identify the "Submission URL". If it is a relative path (like /submit), keep it as is.
            2. Identify the "Question".
            3. Calculate the "Answer". 
               - If the question asks for the email or secret, use the Context Data above.
               - If the question asks to extract a value (like a code or name), extract it exactly.
            
            CRITICAL RULES FOR ANSWERING:
            - If the question asks for "your email", the answer MUST be "{email}".
            - If the question asks for "your secret", the answer MUST be "{student_secret}".
            - Do NOT return the string "your secret". Return the actual value.
            
            Return ONLY a JSON object in this exact format:
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
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            ai_response_text = response.choices[0].message.content
            ai_data = json.loads(ai_response_text)
            
            submission_url = ai_data.get("submission_url")
            payload = ai_data.get("payload")

            # --- FIX: HANDLE RELATIVE URLS ---
            if submission_url and not submission_url.startswith("http"):
                # Combines "https://site.com/page" + "/submit" -> "https://site.com/submit"
                submission_url = urljoin(task_url, submission_url)
            # ---------------------------------

            print(f"    AI Answer: {payload.get('answer')}")
            print(f"    Submitting to: {submission_url}")

            # D. Submit the Answer
            if submission_url:
                submit_response = requests.post(submission_url, json=payload)
                
                # Handle cases where response isn't valid JSON
                try:
                    result = submit_response.json()
                    print(f"    Server Response: {result}")

                    # E. Check for Next Steps (Recursion)
                    if result.get("correct") == True:
                        print("    ✅ Answer Correct!")
                        if "url" in result:
                            next_url = result["url"]
                            print(f"    Found next level: {next_url}")
                            
                            # Close browser to free memory
                            await browser.close()
                            
                            # Recurse
                            await solve_quiz_task(next_url, email, student_secret)
                            return
                    else:
                        print("    ❌ Answer Incorrect or No next URL.")
                except json.JSONDecodeError:
                    print(f"    ❌ Error: Server returned non-JSON response: {submit_response.text}")

        except Exception as e:
            print(f"    Error during processing: {e}")
        
        # Cleanup
        if browser.is_connected():
            await browser.close()

# ==========================================
# 3. THE API ENDPOINT
# ==========================================

@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}