FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:$PATH" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install uv==0.9.7

COPY pyproject.toml uv.lock README.md ./
COPY veriexcite.py streamlit_app.py ./

RUN uv sync --frozen --no-dev

COPY . .

RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /home/appuser/.streamlit \
    && chown -R appuser:appuser /app /home/appuser/.streamlit

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
