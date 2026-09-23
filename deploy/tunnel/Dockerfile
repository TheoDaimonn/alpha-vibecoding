FROM alpine:3.22
RUN apk add --no-cache openssh-client python3
COPY run.py /app/run.py
ENTRYPOINT ["python3", "-u", "/app/run.py"]
