FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src
COPY data/ai_search_mock_dataset.json ./data/ai_search_mock_dataset.json
EXPOSE 8000
CMD ["uvicorn", "ai_search.api:app", "--host", "0.0.0.0", "--port", "8000"]
