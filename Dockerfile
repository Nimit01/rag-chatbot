# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the file with the list of dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We add --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Make port 7860 available to the world outside this container
# This is the default port Gradio runs on
EXPOSE 7860

# Define the command to run your app
# This command runs when the container starts
CMD ["python", "app.py"]