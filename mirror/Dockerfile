FROM python:3.12-slim

WORKDIR /app

COPY mirror/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mirror/app.py .

RUN chown -R nobody:nogroup /app
USER nobody

ENV PORT=8010
EXPOSE 8010

CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
