
FROM python:3.12.10-slim-bookworm

WORKDIR /app

COPY app.py .
COPY util.py .
COPY util2.py .
COPY assets ./assets/
COPY requirements.txt .
COPY .streamlit .streamlit/

RUN mkdir -p artifacts && chmod 777 artifacts
# Non-root user (HF Spaces requirement)
# UID 1000 is the standard user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
