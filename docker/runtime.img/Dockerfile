# Use two stages, on for compile and one for runtime# STAGE 1: compile
# Stage 2: build for runtime
ARG COMPILE_IMAGE
FROM $COMPILE_IMAGE

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Copy over models
RUN mkdir -p "models"
RUN mv /var/models/* models

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 3 --threads 4 --timeout 0 main:app