# Dockerfile

# 1. Use an official, slim Python base image
# Using 3.9 to match your local venv's version
FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 2. Set environment variables for clean logging and behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Silence the "tokenizers" warning we saw before
    TOKENIZERS_PARALLELISM=false

ENV VECTOR_DB_PATH=/app/vector_db_store

# 3. Set a working directory inside the container
WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy your application code AND your vector database
# This copies everything else (app/, core/, scripts/, vector_db_store/, etc.)
COPY --chown=user . .

# 6. Expose the port your app runs on
# EXPOSE 7860

# 7. Define the command to run your app
# This is the same command you use locally
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]