# Use a specific, slim Python version for a smaller image size
FROM python:3.11-slim

# Create a non-root user for better security
RUN useradd -m -u 1000 user
USER user

# Set the working directory and environment path for the new user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

# Set the Hugging Face cache directory to a writable location
ENV HF_HOME /app/cache

# Copy and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# --- ADD THIS COMMAND TO PRE-DOWNLOAD THE MODEL ---
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy all your application files
COPY --chown=user . /app

# Expose the port Gradio will run on
EXPOSE 7860

# The command to run your Gradio app
CMD ["python", "app.py"]