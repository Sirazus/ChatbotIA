# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Create a non-root user
RUN useradd -m -u 1000 user

# Change ownership
RUN chown -R user:user /app

# Switch to the non-root user
USER user

# Expose the port Gunicorn will run on (Using 7860 as in CMD)
EXPOSE 7860

# Command to run the app
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]