FROM python:3.9-slim

WORKDIR /app
COPY requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY client.py /app/client.py

CMD ["python", "client.py"]
