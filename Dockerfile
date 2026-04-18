# ARM64 / Graviton container for AWS Lambda via AWS Lambda Web Adapter.
# The Web Adapter lets us run FastAPI/uvicorn unchanged; it translates
# API Gateway events to HTTP calls against localhost:$PORT.

FROM public.ecr.aws/docker/library/python:3.12-slim AS base

# Inject the Lambda Web Adapter. Harmless outside Lambda — the extension
# only activates when AWS_LAMBDA_RUNTIME_API is set.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /var/task

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-built FAISS cache is produced by CI (or `python build_index.py` locally)
# and baked into the image so cold starts don't hit the network.
COPY .palav_index_cache ./.palav_index_cache
COPY palav ./palav
COPY app.py palav_url_links.txt ./

ENV PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/healthz \
    PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
