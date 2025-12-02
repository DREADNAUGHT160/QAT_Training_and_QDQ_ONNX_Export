# Use an official TensorFlow image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file (updated with TF deps)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directories for outputs
RUN mkdir -p outputs_tf

# Define the command to run the application
CMD ["python", "main_tf.py", "--epochs", "5"]
