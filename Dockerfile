FROM python:3.11-slim

WORKDIR /app

# (valfritt men ofta bra för vissa paket)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiera resten (för "prod"). För dev kör vi ändå volym-mount.
COPY . .

EXPOSE 8888
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
