
FROM python:3.12.10-slim-bookworm

# Non-root user (HF Spaces requirement)
# UID 1000 is the standard user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN mkdir --chown=user -p /app/artifacts/.cache/huggingface

COPY --chown=user app.py .
COPY --chown=user util.py .
COPY --chown=user util2.py .
COPY --chown=user assets ./assets/
COPY --chown=user requirements.txt .
COPY --chown=user .streamlit .streamlit/

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
