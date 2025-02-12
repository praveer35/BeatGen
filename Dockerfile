# Use the official Python image as the base
FROM python:3.12.4

# Set environment variables to prevent buffering
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt update && apt install -y fluidsynth libfluidsynth-dev

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 1601

# Command to run the Flask application
CMD ["python3", "main.py"]