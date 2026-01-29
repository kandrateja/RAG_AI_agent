# ============================================
# RAG AI Agent - Dockerfile
# ============================================
# This builds the FastAPI application with all dependencies:
#   - Docling for document processing
#   - Tesseract OCR (English + Arabic)
#   - PyMuPDF for PDF rendering
#   - AWS Bedrock SDK
#
# Build: docker build -t rag-agent .
# Run:   docker run -p 8000:8000 --env-file .env rag-agent
# ============================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# - gcc, g++: For compiling Python packages
# - postgresql-client: For database health checks
# - libxcb1, libx11-6, etc.: For PyMuPDF/PDF rendering
# - poppler-utils: For PDF processing
# - tesseract-ocr, tesseract-ocr-ara: For OCR (English + Arabic)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    postgresql-client \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libfontconfig1 \
    libfreetype6 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory for document ingestion
RUN mkdir -p /app/uploads && chmod 755 /app/uploads

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
