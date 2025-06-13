FROM python:3.10-slim-buster

WORKDIR /app

# Install system dependencies needed for RDKit drawing
RUN apt-get update && \
    apt-get install -y \
    libcairo2 \
    libpangocairo-1.0-0 \
    libfreetype6 \
    libxrender1 \
    libfontconfig1 \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run your Streamlit app
ENTRYPOINT ["streamlit", "run", "oeb_predictor_app.py"]
