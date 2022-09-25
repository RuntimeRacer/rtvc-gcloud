# Use two stages, on for compile and one for runtime
# STAGE 1: compile
FROM python:3.8.12-bullseye

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Install build dependencies.
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential cmake libeigen3-dev gcc curl libsndfile1 zlib1g-dev

# Setup venv for building all requirements and add it to path
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build and install libwavernn
WORKDIR vocoder/libwavernn
RUN build.sh

# Get and install gcloud SDK to access storage
RUN curl https://sdk.cloud.google.com | bash > /dev/null
ENV PATH="${PATH}:/root/google-cloud-sdk/bin"

# Get Args for model download
ARG STORAGE_KEY
ARG STORAGE_ACCOUNT
ARG MODELS_BUCKET

# Setup the Key and authentication
RUN echo $STORAGE_KEY | base64 --decode > storage-key.json
RUN gcloud auth activate-service-account $STORAGE_ACCOUNT --key-file=storage-key.json

# Get models from gcloud and bundle them in container -> Reduces initial spawn time of the container
RUN mkdir -p /var/models
RUN gsutil -m cp gs://$MODELS_BUCKET/* /var/models