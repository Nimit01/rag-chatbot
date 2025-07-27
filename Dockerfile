# --- Stage 1: The Builder ---
FROM python:3.11-slim as builder

USER root
WORKDIR /app

# Set cache directory
ENV HF_HOME /app/cache

# Install dependencies
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Pre-download the model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# --- Stage 2: The Final Application ---
FROM python:3.11-slim

USER root
WORKDIR /app

# Copy installed packages and model cache from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/cache /app/cache

# Set environment variables for the final stage
ENV HF_HOME /app/cache

# Copy your application code and pre-built database
COPY . .

# Expose the port and run the app
EXPOSE 7860
CMD ["python", "app.py"]