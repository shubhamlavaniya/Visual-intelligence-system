FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements_frontend.txt .
RUN pip install --no-cache-dir -r requirements_frontend.txt

# Copy the frontend application code
COPY frontend/ .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]