# Use two stages, on for compile and one for runtime
# STAGE 1: compile
FROM python:3.8 AS compile-image

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Install build dependencies.
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc

# Setup venv for building all requirements and add it to path
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Get and install gcloud SDK to access storage
RUN curl https://sdk.cloud.google.com | bash > /dev/null
ENV PATH="${PATH}:/root/google-cloud-sdk/bin"

# Get Args for model download
ARG STORAGE_KEY
ARG STORAGE_ACCOUNT
ARG MODELS_BUCKET
ARG ENCODER_MODEL_BUCKET_PATH
ARG SYNTHESIZER_MODEL_BUCKET_PATH
ARG VOCODER_MODEL_BUCKET_PATH

# Setup the Key and authentication
RUN echo $STORAGE_KEY | base64 --decode > storage-key.json
RUN gcloud auth activate-service-account $STORAGE_ACCOUNT --key-file=storage-key.json

# Get models from gcloud and bundle them in container -> Reduces initial spawn time of the container
RUN mkdir -p /var/models
RUN gsutil -m cp -r gs://$MODELS_BUCKET /var/models

# Stage 2: build for runtime
FROM python:3.8 AS build-image

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1

# Copy over venv and enable it
COPY --from=compile-image /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Copy over models
RUN mkdir -p "models"
COPY --from=compile-image /var/models models

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec functions-framework --target=handle_request