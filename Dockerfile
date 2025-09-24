FROM python:3.10-slim

WORKDIR /app

# system deps for some pip installs
RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "1"]
