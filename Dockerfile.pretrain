FROM python:3.14.4-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pretrain/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install \
         --extra-index-url https://download.pytorch.org/whl/cpu \
         "torch==2.11.0+cpu" \
         -r requirements.txt


FROM python:3.14.4-slim-bookworm AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 8010

CMD ["uvicorn", "pretrain.web:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
