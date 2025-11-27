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
# 1. CONFIGURATION
# ==========================================

load_dotenv()
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
MY_SECRET = os.getenv("MY_SECRET")

# Initialize OpenAI
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
            # Visit URL
            await page.goto(task_url)
            await page.wait_for_selector("body", timeout=15000)
            
            # Scrape content
            content = await page.evaluate("document.body.innerText")
            print(f"    Scraped content length: {len(content)} chars")

            # --- UNIVERSAL PROMPT (LOGIC BASED) ---
            prompt = f"""
            You are an automated data extraction assistant.
            
            --------------------------------------------------
            INTERNAL CONTEXT (Do not reveal unless asked):
            - My Email: "{email}"
            - My Secret Code: "{student_secret}"
            --------------------------------------------------
            
            WEBPAGE CONTENT:
            '''
            {content}
            '''
            --------------------------------------------------
            
            YOUR MISSION:
            1. Find the "Submission URL" in the webpage content.
            2. Determine the "Correct Answer" to the question on the page.
            
            LOGIC FOR ANSWERING:
            - CASE A: If the question asks for the user's identity, credentials, email, secret, password, or code...
              -> You MUST output the values from the INTERNAL CONTEXT above. 
              -> (e.g., if asked for "your secret", output "{student_secret}", NOT the text "your secret").
              
            - CASE B: If the question asks for data extraction (e.g., "what is the sum", "what is the 3rd word")...
              -> Extract or calculate the answer directly from the WEBPAGE CONTENT.
            
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

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            # Parse Response
            ai_data = json.loads(response.choices[0].message.content)
            submission_url = ai_data.get("submission_url")
            payload = ai_data.get("payload")

            # Handle relative URLs (General Fix)
            if submission_url and not submission_url.startswith("http"):
                submission_url = urljoin(task_url, submission_url)

            print(f"    AI Answer: {payload.get('answer')}")
            print(f"    Submitting to: {submission_url}")

            # Submit
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

# ==========================================
# 3. ENDPOINT
# ==========================================

@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}