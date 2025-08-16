# Dockerfile (sample) for Flask + Gunicorn
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (add more as needed)
#RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    build-essential curl \
    libcairo2 \
    libpango-1.0-0 libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    fonts-dejavu-core \
    shared-mime-info \
  && rm -rf /var/lib/apt/lists/*


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source last for better caching during CI
COPY . .

# Expose port (must match compose and CI)
EXPOSE 8000

# Adjust 'app:app' to your Flask WSGI module and variable
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]