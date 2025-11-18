FROM python:3.9-slim

WORKDIR /app
COPY requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only lightweight server wrapper; mount code via volumes at runtime
COPY server.py /app/server.py

EXPOSE 5000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]
