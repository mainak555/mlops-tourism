
FROM python:3.12.10-slim-bookworm

# Non-root user (HF Spaces requirement)
# UID 1000 is the standard user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade mlflow pymysql

EXPOSE 7860

CMD mlflow server \
    --host 0.0.0.0 \
    --port 7860 \
    --allowed-hosts "*" \
    --cors-allowed-origins "*" \
    --serve-artifacts
