import os
import json
import time
import base64
import re
from urllib.parse import urljoin, urlparse

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from openai import OpenAI

# ==========================================
# CONFIG
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
# UTILITIES
# ==========================================
def mask_secret(s: str) -> str:
    """Masks secret for LOGGING only. Do not use for the actual Prompt."""
    if not s:
        return ""
    return s[:2] + "***" if len(s) > 3 else "***"

def extract_numbers_from_text(text: str):
    """
    Extract ints/floats from text (handles thousands separators).
    """
    if not text:
        return []
    # Remove commas used as thousands separators
    cleaned = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)
    # Find numbers (integers or floats)
    tokens = re.findall(r'-?\d+(?:\.\d+)?', cleaned)
    nums = []
    for t in tokens:
        try:
            if '.' in t:
                nums.append(float(t))
            else:
                nums.append(int(t))
        except:
            pass
    return nums

def clean_json_string(s: str) -> str:
    """Removes markdown code blocks if the AI adds them."""
    if "```json" in s:
        s = s.split("```json")[1].split("```")[0]
    elif "```" in s:
        s = s.split("```")[1].split("```")[0]
    return s.strip()

def page_looks_for_secret(page_text: str, evidence: str) -> bool:
    """
    Heuristic: Checks if the page is explicitly asking for credentials.
    """
    s = (page_text or "") + "\n" + (evidence or "")
    s = s.lower()
    keywords = [
        "enter your secret", "your secret", "student secret", "enter secret",
        "secret code", "provide secret", "enter code", "password:", "auth code"
    ]
    if any(kw in s for kw in keywords):
        return True
    return False

# ==========================================
# FILE PROCESSING TOOLS
# ==========================================
async def process_audio_url(url: str) -> str:
    """
    Try multipart upload first (best for proxies), then JSON base64 fallback.
    """
    print(f"    [Tool] 🔊 Found Audio: {url}")
    try:
        resp = requests.get(url, timeout=20)
        if 'text/html' in resp.headers.get('Content-Type', ''):
            print("    [Tool] Audio URL returned HTML, skipping.")
            return ""

        audio_bytes = resp.content
        headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}"}

        # Attempt 1: Multipart (model param in query + data) - Most Robust for AI Pipe
        transcription_url = "https://aipipe.org/openai/v1/audio/transcriptions?model=whisper-1"
        files = {"file": ("audio.opus", audio_bytes, "application/octet-stream")}
        data = {"model": "whisper-1"}

        print("    [Tool] Whisper Attempt 1: Multipart")
        r = requests.post(transcription_url, headers=headers, files=files, data=data, timeout=60)

        if r.status_code == 200:
            text = r.json().get("text", "")
            print(f"    [Tool] Transcription OK: {text[:80]}...")
            return text

        # Attempt 2: JSON with base64 file (Fallback)
        print("    [Tool] Whisper Attempt 1 failed; Attempt 2: JSON base64")
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        payload = {
            "model": "whisper-1", 
            "file_b64": b64, 
            "filename": "audio.mp3"
        }

        r2 = requests.post("https://aipipe.org/openai/v1/audio/transcriptions", headers=headers, json=payload, timeout=60)

        if r2.status_code == 200:
            text = r2.json().get("text", "")
            print(f"    [Tool] Transcription OK (base64): {text[:80]}...")
            return text

        print(f"    [Tool] All Whisper attempts failed. Status: {r.status_code} / {r2.status_code}")
        return ""

    except Exception as e:
        print("    [Tool] Audio Exception:", e)
        return ""


async def process_csv_url(url: str) -> str:
    print(f"    [Tool] 📊 Found CSV/Data: {url}")
    try:
        r = requests.get(url, timeout=20)
        if 'text/html' in r.headers.get('Content-Type', ''):
            print("    [Tool] CSV URL returned HTML, skipping.")
            return ""
        content = r.content.decode("utf-8", errors="replace")
        print(f"    [Tool] CSV length: {len(content)} chars")
        return content
    except Exception as e:
        print("    [Tool] CSV Error:", e)
        return ""

# ==========================================
# MAIN SOLVER
# ==========================================
async def solve_quiz_task(task_url: str, email: str, student_secret: str):
    if not task_url:
        return
    print(f"\n[+] Processing Task: {task_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(task_url)
            await page.wait_for_selector("body", timeout=15000)

            page_text = await page.evaluate("document.body.innerText")
            
            # Extract links and media sources
            media_links = await page.evaluate("""() => {
                let links = [];
                document.querySelectorAll('a').forEach(a => { if (a.href) links.push({url: a.href, type: 'link'}) });
                document.querySelectorAll('audio, source').forEach(el => { if (el.src) links.push({url: el.src, type: 'audio'}) });
                return links;
            }""")

            evidence_log = ""
            processed_urls = set()
            csv_contents = ""
            audio_transcripts = []

            # Download and Process Files
            for item in media_links:
                url = item['url']
                if not url or url in processed_urls or url.rstrip('/') == task_url.rstrip('/'):
                    continue

                processed_urls.add(url)
                
                # Check for Audio
                if url.lower().endswith(('.mp3', '.wav', '.opus')) or item['type'] == 'audio':
                    t = await process_audio_url(url)
                    if t:
                        audio_transcripts.append(t)
                        evidence_log += f"\n[AUDIO TRANSCRIPT FROM {url}]\n{t}\n"
                
                # Check for CSV
                elif url.lower().endswith('.csv') or ('csv' in url.lower() and 'download' in url.lower()):
                    c = await process_csv_url(url)
                    if c:
                        csv_contents += c
                        evidence_log += f"\n[CSV FILE CONTENT FROM {url}]\n{c}\n"

            # -------------------------
            # AI DECISION
            # -------------------------
            system_msg = (
                "You are a strict JSON quiz-solving agent. Output valid JSON only."
            )

            # NOTE: We send the ACTUAL secret to the AI so it can output it if asked.
            # We rely on the Prompt Instructions to tell it when to use it.
            user_msg = f"""
INTERNAL DATA:
EMAIL: {email}
SECRET: {student_secret}

EVIDENCE:
{evidence_log}

PAGE TEXT:
{page_text}

TASK:
1. Find the "submission_url" on the page.
2. Answer the question.

RULES:
- IDENTITY: If asked for "your secret", "password", or "email", use the INTERNAL DATA values.
- SCRAPING: If the text says "The secret code is XYZ", use "XYZ".
- MATH: If asked to sum numbers, use the EVIDENCE data.

Return valid JSON:
{{ "submission_url": "...", "payload": {{ "email": "{email}", "secret": "{student_secret}", "url": "{task_url}", "answer": <value> }} }}
"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
            except Exception as e:
                print("    [AI] Model call error:", e)
                response = None

            ai_payload = {}
            submission_url = None
            
            if response:
                raw_content = response.choices[0].message.content
                try:
                    # Clean markdown if present
                    clean_raw = clean_json_string(raw_content)
                    ai_json = json.loads(clean_raw)
                    submission_url = ai_json.get("submission_url")
                    ai_payload = ai_json.get("payload", {})
                except Exception:
                    print("    [AI] JSON Parse Error. Raw:", raw_content[:100])

            # Prepare Payload
            payload = {
                "email": email,
                "secret": student_secret,
                "url": task_url,
                "answer": ai_payload.get("answer")
            }

            # -------------------------
            # MATH VERIFICATION (Local)
            # -------------------------
            # Only use CSV numbers if CSV is present to avoid summing page dates/garbage.
            numbers_in_csv = extract_numbers_from_text(csv_contents) if csv_contents else []
            numbers_in_audio = []
            for t in audio_transcripts:
                numbers_in_audio.extend(extract_numbers_from_text(t))
            
            # Check for calculation keywords
            calc_keywords = r'\b(sum|total|add|calculate|count)\b'
            is_calculation = re.search(calc_keywords, page_text.lower())

            used_local_calc = False
            
            # Logic: If CSV exists and calculation is asked, sum CSV.
            if numbers_in_csv and is_calculation:
                local_sum = sum(numbers_in_csv)
                payload["answer"] = int(local_sum) if isinstance(local_sum, float) and local_sum.is_integer() else local_sum
                used_local_calc = True
                print(f"    [Verify] Overriding AI with CSV Sum: {payload['answer']}")
            
            # Fallback: If no CSV but Audio exists and calculation is asked, sum Audio numbers.
            elif not numbers_in_csv and numbers_in_audio and is_calculation:
                local_sum = sum(numbers_in_audio)
                payload["answer"] = int(local_sum) if isinstance(local_sum, float) and local_sum.is_integer() else local_sum
                used_local_calc = True
                print(f"    [Verify] Overriding AI with Audio Sum: {payload['answer']}")

            # -------------------------
            # SECRET VERIFICATION (Safety Net)
            # -------------------------
            # If AI returned a placeholder text like "your secret", swap it manually.
            raw_ans_str = str(payload["answer"]).lower()
            placeholders = ["your secret", "my secret", "student secret", "anything you want"]
            
            if any(ph in raw_ans_str for ph in placeholders):
                if page_looks_for_secret(page_text, evidence_log):
                    print("    ⚠️ AI returned placeholder. Swapping with actual secret.")
                    payload["answer"] = student_secret

            # URL Cleanup
            if submission_url:
                if not str(submission_url).startswith("http"):
                    submission_url = urljoin(task_url, submission_url)
            else:
                # Default fallback
                parsed = urlparse(task_url)
                submission_url = f"{parsed.scheme}://{parsed.netloc}/submit"

            payload["url"] = task_url
            
            print(f"    Final Answer: {payload.get('answer')}")
            print(f"    Submitting to: {submission_url}")

            # -------------------------
            # SUBMIT
            # -------------------------
            try:
                submit_res = requests.post(submission_url, json=payload, timeout=30)
                res_json = submit_res.json()
                print("    Server:", res_json)

                if res_json.get("correct") == True:
                    print("    ✅ Correct!")
                    if res_json.get("url"):
                        await browser.close()
                        await solve_quiz_task(res_json["url"], email, student_secret)
                elif res_json.get("url"):
                    print("    ❌ Incorrect. Retrying next URL...")
                    await browser.close()
                    await solve_quiz_task(res_json["url"], email, student_secret)
                else:
                    print("    🛑 Game Over.")
            except Exception as e:
                print("    ❌ Submit Error:", e)

        except Exception as e:
            print("    Page Error:", e)

        if browser.is_connected():
            await browser.close()

# ==========================================
# API
# ==========================================
@app.post("/run")
async def run_task(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    background_tasks.add_task(solve_quiz_task, request.url, request.email, request.secret)
    return {"message": "Task started", "status": "processing"}