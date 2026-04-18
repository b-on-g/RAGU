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

COPY etl/ ./etl/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir pymssql python-dotenv sentence-transformers

COPY server/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8000

CMD ["./entrypoint.sh"]
