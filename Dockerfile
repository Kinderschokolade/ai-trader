FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    AITRADER_HOME=/app \
    MPLBACKEND=Agg

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[llm]"

# config/, data/ and results/ are mounted from the host (see docker-compose.yml)
ENTRYPOINT ["aitrader"]
CMD ["--help"]
