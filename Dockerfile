FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3" \
    && pip install -r requirements.txt
COPY src ./src
COPY ru_pii ./ru_pii
COPY artifacts/student-pii.pt ./artifacts/student-pii.pt
EXPOSE 8000
CMD ["python", "-m", "src.run_api"]
