# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Get Args for build job
ARG STORAGE_KEY
ARG STORAGE_ACCOUNT
ARG MODELS_BUCKET
ARG ENCODER_MODEL_BUCKET_PATH
ARG SYNTHESIZER_MODEL_BUCKET_PATH
ARG VOCODER_MODEL_BUCKET_PATH

# Small test to check build args are working
RUN echo $MODELS_BUCKET

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1
RUN pip install --no-cache-dir -r requirements.txt

# Get and install gcloud SDK to access storage
RUN curl https://sdk.cloud.google.com | bash > /dev/null
ENV PATH="${PATH}:/root/google-cloud-sdk/bin"

# Setup the Key and authentication
RUN echo $STORAGE_KEY | base64 --decode > storage-key.json
RUN gcloud auth activate-service-account $STORAGE_ACCOUNT --key-file=storage-key.json

# Get models from gcloud and bundle them in container -> Reduces initial spawn time of the container
RUN mkdir -p "models"
RUN gsutil cp gs://$MODELS_BUCKET/$ENCODER_MODEL_BUCKET_PATH "models/encoder.pt"
RUN gsutil cp gs://$MODELS_BUCKET/$SYNTHESIZER_MODEL_BUCKET_PATH "models/synthesizer.pt"
RUN gsutil cp gs://$MODELS_BUCKET/$VOCODER_MODEL_BUCKET_PATH "models/vocoder.pt"

# Cleanup; shrink the image
RUN rm storage-key.json && rm -rf /root/google-cloud-sdk/ && apt-get autoremove

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec functions-framework --target=handle_request