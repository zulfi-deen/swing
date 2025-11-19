# Multi-stage build for Swing Trading System

# Stage 1: Python API
FROM python:3.11-slim as api

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Expose API port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 2: React UI (optional, for production build)
FROM node:18-alpine as ui

WORKDIR /app

COPY src/ui/package*.json ./
RUN npm install

COPY src/ui/ ./
RUN npm run build

# Stage 3: Final (combine if needed)
FROM nginx:alpine

COPY --from=ui /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

