FROM python:3.9-slim

WORKDIR /app

COPY securebank /app/securebank

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir -r /app/securebank/requirements.txt

EXPOSE 5000

CMD ["python3", "-m", "securebank.app"]