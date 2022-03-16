# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1
RUN pip install --no-cache-dir -r requirements.txt

# Get and install gcloud SDK to access storage
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Get models from gcloud and bundle them in container -> Reduces init time
gcloud auth login
gsutil cp gs://$MODELS_BUCKET/$ENCODER_MODEL_BUCKET_PATH $ENCODER_MODEL_LOCAL_PATH
gsutil cp gs://$MODELS_BUCKET/$SYNTHESIZER_MODEL_BUCKET_PATH $SYNTHESIZER_MODEL_LOCAL_PATH
gsutil cp gs://$MODELS_BUCKET/$VOCODER_MODEL_BUCKET_PATH $VOCODER_MODEL_LOCAL_PATH

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec functions-framework --target=handle_request