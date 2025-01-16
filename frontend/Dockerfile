FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 cholera && chown -R cholera:cholera /app
USER cholera

# Expose port
EXPOSE 8050

# Default command
CMD ["python", "app.py"]