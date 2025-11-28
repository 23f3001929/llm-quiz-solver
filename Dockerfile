FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

WORKDIR /app

# Install ffmpeg (Required for Audio Processing)
RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]