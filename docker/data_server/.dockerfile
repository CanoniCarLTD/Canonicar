FROM ros:humble

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy server code
COPY . /app/
WORKDIR /app

# Expose WebSocket port
EXPOSE 8766

# Start the server
CMD ["python3", "data_server.py"]