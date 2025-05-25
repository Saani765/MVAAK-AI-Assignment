FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure your ONNX model file (e.g., stt_hi_conformer_ctc_medium.onnx) is in the same directory
# as your main.py or adjust the COPY command and ONNX_MODEL_PATH in main.py accordingly.

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 