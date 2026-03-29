# --- Frontend ---
FROM node:20-alpine AS frontend

RUN apk add --no-cache git

WORKDIR /app
RUN git clone --depth 1 https://github.com/hyoo-ru/mam.git . \
    && npm install

COPY front/ bog/RAGU/front/

RUN npx mam bog/RAGU/front/app

EXPOSE 9080

CMD ["npm", "start"]


# --- API ---
FROM python:3.12-slim AS api

WORKDIR /app

COPY pyproject.toml ./
COPY ragu/ ./ragu/
RUN pip install --no-cache-dir .

COPY server/ ./server/
RUN pip install --no-cache-dir -r server/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
