# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set a working directory in the container
WORKDIR /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Install WhisperX from GitHub (and its dependencies)
RUN pip install git+https://github.com/m-bain/whisperX.git

# (Optional) Install any extra dependencies your predict.py might need
RUN pip install requests

# Copy your prediction script into the container
COPY predict.py /app/predict.py

# The command that Replicate will use to run inference
CMD ["python", "predict.py"]
