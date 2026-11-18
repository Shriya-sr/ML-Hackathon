# Optional container for the Laptop 1 API. Not required for the demo (see docs/deployment.md).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY backend ./backend
COPY configs ./configs
COPY artifacts ./artifacts
RUN pip install --no-cache-dir -e .
ENV PATHSNAP_HOST=0.0.0.0 PATHSNAP_PORT=8000 PATHSNAP_DATA_DIR=/data
EXPOSE 8000
CMD ["python", "-m", "backend.main"]
