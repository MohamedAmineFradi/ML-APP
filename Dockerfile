# Use official Python runtime as base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all application code to container
COPY . .

RUN  mkdir -p models

# Specify default command to run the application (customize as needed)
CMD ["python", "src/train.py"]
