import os
import json
import asyncio
import requests
import time
import base64
import re
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from openai import OpenAI

# =====================================================
# 1. CONFIGURATION
# =====================================================

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


# =====================================================
# 2. UTILITIES
# =====================================================

def extract_numbers_from_text(text: str):
    cleaned = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)
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

def looks_like_placeholder(answer: str):
    if not answer:
        return True
    s = str(answer).lower()
    placeholders = [
        "anything you want",
        "your secret",
        "my secret",
        "the secret code",
        "the secret code you scraped",
        "scraped",
        "placeholder",
        "your email"
    ]
    return any(ph in s for ph in placeholders)


# =====================================================
# 3. FILE PROCESSING TOOLS
# =====================================================

async def process_audio_url(url: str) -> str:
    print(f"    [Tool] 🔊 Found Audio: {url}")
    try:
        resp = requests.get(url, timeout=15)
        if 'text/html' in resp.headers.get('Content-Type', ''):
            return ""

        audio_bytes = resp.content
        headers = { "Authorization": f"Bearer {AIPIPE_TOKEN}" }

        # First attempt: multipart
        print("    [Tool] Whisper Attempt 1: Multipart")
        transcription_url = "https://aipipe.org/openai/v1/audio/transcriptions?model=whisper-1"
        files = {"file": ("audio.opus", audio_bytes, "audio/opus")}
        data = {"model": "whisper-1"}

        r = requests.post(transcription_url, headers=headers, files=files, data=data, timeout=60)

        if r.status_code == 200:
            text = r.json().get("text", "")
            print(f"    [Tool] Transcription OK: {text[:70]}")
            return f"\n[AUDIO TRANSCRIPT FROM {url}]\n{text}\n"

        # fallback
        print("    [Tool] Whisper Attempt 1 Failed. Attempt 2: JSON Base64")
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        payload = {
            "model": "whisper-1",
            "file_b64": b64,
            "filename": "audio.opus"
        }

        r2 = requests.post(
            "https://aipipe.org/openai/v1/audio/transcriptions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if r2.status_code == 200:
            text = r2.json().get("text", "")
            print(f"    [Tool] Transcription OK (fallback): {text[:70]}")
            return f"\n[AUDIO TRANSCRIPT FROM {url}]\n{text}\n"

        print(f"    [Tool] Whisper Errors: {r.text[:200]} | {r2.text[:200]}")
        return ""

    except Exception as e:
        print("    [Tool] Audio Exception:", e)
        return ""


async def process_csv_url(url: str) -> str:
    print(f"    [Tool] 📊 Found CSV/Data: {url}")
    try:
        r = requests.get(url, timeout=15)
        if 'text/html' in r.headers.get('Content-Type', ''): 
            return ""
        content = r.content.decode("utf-8")
        return f"\n[CSV FILE CONTENT FROM {url}]\n{content}\n"
    except Exception as e:
        print("    [Tool] CSV Error:", e)
        return ""


# =====================================================
# 4. MAIN SOLVER LOGIC
# =====================================================

async def solve_quiz_task(task_url: str, email: str, student_secret: str):
    print(f"\n[+] Processing Task: {task_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(task_url)
            await page.wait_for_selector("body", timeout=15000)

            page_text = await page.evaluate("document.body.innerText")

            media_links = await page.evaluate("""() => {
                let arr = [];
                document.querySelectorAll('a').forEach(a => arr.push({url: a.href, type: "link"}));
                document.querySelectorAll('audio, source').forEach(a => {
                    if (a.src) arr.push({url: a.src, type: "audio"});
                });
                return arr;
            }""")

            evidence = ""
            used = set()

            for item in media_links:
                url = item["url"]
                if url in used:
                    continue

                is_audio = url.endswith(".mp3") or url.endswith(".wav") or url.endswith(".opus")
                is_csv = url.endswith(".csv")

                if is_audio:
                    used.add(url)
                    evidence += await process_audio_url(url)
                elif is_csv:
                    used.add(url)
                    evidence += await process_csv_url(url)

            # ------------------------------
            # LLM prompt (deterministic)
            # ------------------------------
            system_msg = (
                "You are a strict JSON quiz-solving agent. "
                "You MUST output valid JSON only. "
                "Keys: submission_url, payload(email,secret,url,answer)."
            )

            user_msg = f"""
INTERNAL DATA:
EMAIL={email}
SECRET={student_secret}

EVIDENCE:
{evidence}

PAGE TEXT:
{page_text}

Return ONLY valid JSON:
{{
  "submission_url": "...",
  "payload": {{
     "email": "{email}",
     "secret": "{student_secret}",
     "url": "{task_url}",
     "answer": <answer_value>
  }}
}}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )

            raw = response.choices[0].message.content
            try:
                ai = json.loads(raw)
            except:
                print("    JSON parse failed. Raw:", raw)
                return

            submission_url = ai.get("submission_url")
            payload = ai.get("payload", {})

            # ------------------------------
            # Safety: fix placeholders
            # ------------------------------
            ans = payload.get("answer")
            if looks_like_placeholder(ans):
                print("    ⚠️ Placeholder detected → replacing with actual student secret")
                payload["answer"] = student_secret

            # ------------------------------
            # Local numeric verification
            # ------------------------------
            combined = evidence + "\n" + page_text
            nums = extract_numbers_from_text(combined)

            if nums:
                # Detect sum type question
                if ("sum" in page_text.lower()) or ("total" in page_text.lower()):
                    local_sum = sum(nums)
                    try:
                        model_ans = float(str(payload["answer"]))
                        if abs(model_ans - local_sum) > 1e-6:
                            print("    ⚠️ Model answer incorrect → using local sum:", local_sum)
                            payload["answer"] = int(local_sum) if local_sum.is_integer() else local_sum
                    except:
                        payload["answer"] = int(local_sum) if local_sum.is_integer() else local_sum

            # Fix URLs
            if submission_url and not submission_url.startswith("http"):
                submission_url = urljoin(task_url, submission_url)
            payload["url"] = task_url

            print("    Final Answer:", payload["answer"])
            print("    Submitting to:", submission_url)

            # ------------------------------
            # Submit to server
            # ------------------------------
            r = requests.post(submission_url, json=payload)
            try:
                res_json = r.json()
                print("    Server:", res_json)

                if res_json.get("correct"):
                    print("    ✅ Correct!")
                    if res_json.get("url"):
                        await browser.close()
                        await solve_quiz_task(res_json["url"], email, student_secret)

                elif res_json.get("url"):
                    print("    ❌ Incorrect → continuing next level")
                    await browser.close()
                    await solve_quiz_task(res_json["url"], email, student_secret)

                else:
                    print("    ❌ Game Over.")

            except Exception:
                print("    Submit Error:", r.text)

        except Exception as e:
            print("    Error:", e)

        if browser.is_connected():
            await browser.close()


# =====================================================
# 5. API ENDPOINT
# =====================================================

@app.post("/run")
async def run_task(req: QuizRequest, background: BackgroundTasks):
    if req.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    background.add_task(solve_quiz_task, req.url, req.email, req.secret)
    return {"message": "Task started", "status": "processing"}

