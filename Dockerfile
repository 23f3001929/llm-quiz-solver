# Use the stable Playwright image
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

# Set work directory
WORKDIR /app

# CRITICAL: Install ffmpeg for audio conversion
RUN apt-get update && apt-get install -y ffmpeg

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Start the server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]