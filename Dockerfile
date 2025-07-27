# --- Stage 1: The Builder ---
# This stage installs all dependencies and downloads the model
FROM python:3.11-slim as builder

# Set up user and environment
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app
ENV HF_HOME /app/cache

# Copy only the requirements file and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Download the sentence transformer model into the cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# --- Stage 2: The Final Application ---
# This stage creates the final, clean image
FROM python:3.11-slim

# Set up the same user
RUN useradd -m -u 1000 user
USER user
WORKDIR /app

# Copy the installed packages and the model cache from the builder stage
COPY --chown=user --from=builder /home/user/.local /home/user/.local
COPY --chown=user --from=builder /app/cache /app/cache

# Set the environment variables again for the final stage
ENV PATH="/home/user/.local/bin:$PATH"
ENV HF_HOME /app/cache

# Copy your application code and data
COPY --chown=user . /app

# Expose the port and run the app
EXPOSE 7860
CMD ["python", "app.py"]