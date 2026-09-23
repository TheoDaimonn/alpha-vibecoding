FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JUPYTER_PORT=9132

WORKDIR /workspace

ENV CERTS=/usr/local/share/ca-certificates/vimpelcom-int-ca.crt
COPY certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates
ARG USER=appuser
ARG UID=1001
ARG GID=1001
RUN groupadd --gid ${GID} ${USER} \
    && useradd  --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USER}
RUN chown -R ${USER}:${USER} /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install \
    --timeout 300 \
    --retries 10 \
    --no-cache-dir \
    -r /tmp/requirements.txt

EXPOSE 9132

CMD ["jupyter", "lab", "--ip", "0.0.0.0", "--port", "9132", "--no-browser"]
