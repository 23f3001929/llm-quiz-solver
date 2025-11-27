import os
import json
import asyncio
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from openai import OpenAI

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

# Load secrets from the .env file
load_dotenv()

# Get secrets from environment variables
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
MY_SECRET = os.getenv("MY_SECRET")

# Check if keys are loaded (Safety check)
if not AIPIPE_TOKEN or not MY_SECRET:
    raise ValueError("Missing secrets! Make sure you created a .env file with AIPIPE_TOKEN and MY_SECRET.")

# Initialize OpenAI Client to use AI Pipe
client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url="https://aipipe.org/openai/v1"
)

app = FastAPI()

# Data Model for the incoming request from the evaluator
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# ==========================================
# 2. THE SOLVER LOGIC
# ==========================================

async def solve_quiz_task(task_url: str, email: str, student_secret: str):
    """
    Scrapes the page, asks LLM for the answer, submits it, 
    and recurses if a new URL is found.
    """
    print(f"\n[+] Processing Task: {task_url}")
    
    async with async_playwright() as p:
        # Launch browser (Headless = invisible)
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            # A. Visit the URL
            await page.goto(task_url)
            
            # Wait for body to ensure JS has rendered the content
            await page.wait_for_selector("body", timeout=15000)
            
            # B. Scrape the text
            content = await page.evaluate("document.body.innerText")
            print(f"    Scraped content length: {len(content)} chars")

            # C. AI Analysis
            # We ask the AI to return a specific JSON structure we can use immediately
            prompt = f"""
            You are a data extraction agent. 
            Analyze the following text from a quiz page:
            ----------------
            {content}
            ----------------
            
            Extract:
            1. The "Submission URL" (where to POST the answer).
            2. The "Answer" to the question asked in the text.
            
            The user's email is: "{email}"
            The user's secret is: "{student_secret}"
            The current page URL is: "{task_url}"
            
            Return ONLY a JSON object in this exact format:
            {{
                "submission_url": "https://...",
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
                response_format={"type": "json_object"} # Force valid JSON
            )

            # Parse the AI's JSON response
            ai_response_text = response.choices[0].message.content
            ai_data = json.loads(ai_response_text)
            
            submission_url = ai_data.get("submission_url")
            payload = ai_data.get("payload")

            print(f"    AI Answer: {payload.get('answer')}")
            print(f"    Submitting to: {submission_url}")

            # D. Submit the Answer
            submit_response = requests.post(submission_url, json=payload)
            result = submit_response.json()
            
            print(f"    Server Response: {result}")

            # E. Check for Next Steps (Recursion)
            if result.get("correct") == True:
                print("    ✅ Answer Correct!")
                if "url" in result:
                    next_url = result["url"]
                    print(f"    Found next level: {next_url}")
                    
                    # Close current browser before starting next task to save memory
                    await browser.close()
                    
                    # Recursive call to solve the next page
                    await solve_quiz_task(next_url, email, student_secret)
                    return
            else:
                print("    ❌ Answer Incorrect or No next URL.")

        except Exception as e:
            print(f"    Error during processing: {e}")
        
        # Ensure browser is closed if we didn't recurse
        if browser.is_connected():
            await browser.close()

# ==========================================
# 3. THE API ENDPOINT
# ==========================================

@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    # Step A: Verify the secret
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    # Step B: Start the solver in the background
    # We pass the email and secret so the solver can use them in the answer payload
    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    
    return {"message": "Task started", "status": "processing"}

# To run: uvicorn main:app --reload --port 8000