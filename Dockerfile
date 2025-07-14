FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY dna_predictor_app/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the src directory
COPY src ./src

# Copy the data directory
COPY data ./data

# Copy application files
COPY dna_predictor_app ./dna_predictor_app

COPY Results ./Results

WORKDIR /app/dna_predictor_app

EXPOSE 5000

CMD ["python", "app.py"]
