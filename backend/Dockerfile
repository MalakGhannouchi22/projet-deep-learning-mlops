FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY backend ./backend
RUN pip install --no-cache-dir -e ".[ml]"

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
