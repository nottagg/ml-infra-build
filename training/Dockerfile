FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    unzip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY kaggle.json /root/.kaggle/kaggle.json
COPY train.py /app/train.py 
COPY start.sh /app/start.sh

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Set permission for kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json
RUN chmod +x /app/start.sh

# Run training script
CMD ["/app/start.sh"]
