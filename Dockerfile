FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install early
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create writable cache directory
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# Copy app files
COPY . .

# Install new langchain-huggingface package
RUN pip install -U langchain-huggingface

# Expose default Gradio port
EXPOSE 7860

# Launch app
CMD ["python", "app.py"]
