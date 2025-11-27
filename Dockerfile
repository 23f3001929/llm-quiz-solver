
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy


WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Run the application using the PORT variable provided by Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]# UPDATE: Using the newer version 1.56.0 as requested by the error
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Run the application using the PORT variable provided by Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]