FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["python", "-m", "src.data.clean_data"]
