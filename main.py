import os
import pathlib

import render
import flask
import base64
import logging
import torch
import numpy as np

from time import perf_counter as timer
from google.cloud import storage
from google.cloud import logging as g_log

# SV2TTS
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# Env vars
MODELS_BUCKET = os.environ['MODELS_BUCKET']
ENCODER_MODEL_BUCKET_PATH = os.environ['ENCODER_MODEL_BUCKET_PATH']
ENCODER_MODEL_LOCAL_PATH = os.environ['ENCODER_MODEL_LOCAL_PATH']
SYNTHESIZER_MODEL_BUCKET_PATH = os.environ['SYNTHESIZER_MODEL_BUCKET_PATH']
SYNTHESIZER_MODEL_LOCAL_PATH = os.environ['SYNTHESIZER_MODEL_LOCAL_PATH']
VOCODER_MODEL_BUCKET_PATH = os.environ['VOCODER_MODEL_BUCKET_PATH']
VOCODER_MODEL_LOCAL_PATH = os.environ['VOCODER_MODEL_LOCAL_PATH']

# Cloud Function related stuff
app = flask.Flask(__name__)
if MODELS_BUCKET != "LOCAL":
    log_client = g_log.Client()
    log_client.setup_logging(log_level=logging.INFO)

# Hello World simple test function
def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    if request.args and 'message' in request.args:
        return request.args.get('message')
    elif request_json and 'message' in request_json:
        return request_json['message']
    else:
        return f'Hello World!'


# entry point of this gcloud function
# Expects requests to be sent via POST method.
# In case of a non-POST Request it will respond with get_version().
@app.route("/")
@app.route("/encode")
@app.route("/synthesize")
@app.route("/vocode")
@app.route("/render")
def handle_request(request: flask.Request):
    # Parse request data anyways
    method = request.method
    request_data = request.get_json()
    if method != 'POST' or not request_data:
        if request.args:
            return flask.make_response(get_version(request))
        elif request_data:
            return flask.make_response(get_version(request))
        else:
            return flask.make_response(get_version(request))

    # Get route and forward request
    if 'encode' in request.path:
        return process_encode_request(request_data)
    if 'synthesize' in request.path:
        return process_synthesize_request(request_data)
    if 'vocode' in request.path:
        return process_vocode_request(request_data)
    if 'render' in request.path:
        return process_render_request(request_data)
    else:
        return flask.make_response(get_version(request))

# process_encode_request
# Input params:
# - binary wav file of speaker voice
# Returns:
# - speaker embedding generated from input wav
# - embedding graph generated from input wav
# - mel spectogram graph generated from input wav
def process_encode_request(request_data):
    # Gather data from request
    wav = request_data["speaker_wav"] if "speaker_wav" in request_data else None

    if wav is None:
        return error_response("no speaker wav provided")

    # Generate the spectogram - if this fails, audio data provided is invalid
    try:
        # Decode the wav from payload
        wav = base64.b64decode(wav)
        # Generate the spectogram
        spectogram = Synthesizer.make_spectrogram(wav)
    except Exception as e:
        logging.log(logging.ERROR, e)
        return error_response("invalid speaker wav provided")

    # Download speaker encoder model from storage bucket
    if MODELS_BUCKET != "LOCAL" and not os.path.exists(ENCODER_MODEL_LOCAL_PATH):
        storage_client = storage.Client()
        encoders_bucket = storage_client.get_bucket(MODELS_BUCKET)
        encoder_model = encoders_bucket.blob(ENCODER_MODEL_BUCKET_PATH)
        encoder_model.download_to_filename(ENCODER_MODEL_LOCAL_PATH)

    # Load Speaker encoder
    if os.path.exists(ENCODER_MODEL_LOCAL_PATH):
        start = timer()
        encoder.load_model(pathlib.Path(ENCODER_MODEL_LOCAL_PATH))
        logging.log(logging.INFO, "Successfully loaded encoder (%dms)." % int(1000 * (timer() - start)))
    else:
        return error_response("encoder model not found", 500)

    # process wav and generate embedding
    encoder_wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(encoder_wav)

    # Build response
    response = {
        "embed": base64.b64encode(embed).decode('utf-8'),
        "embed_graph": render.embedding(embed),
        "embed_mel": render.spectogram(spectogram)
    }

    return flask.make_response(response)

# process_synthesize_request
# Input params:
# - speaker embedding
# - text to synthesize via embedding
# (- fixed neural network seed)
# Returns:
# - binary spectogram of synthesized text
# - mel spectogram graph generated from output spectogram
def process_synthesize_request(request_data):
    # Gather data from request
    embed = request_data["speaker_embed"] if "speaker_embed" in request_data else None
    text = request_data["text"] if "text" in request_data else None
    seed = request_data["seed"] if "seed" in request_data else None

    # Check input
    if embed is None:
        return error_response("no speaker embedding provided")
    if text is None or len(text) < 1:
        return error_response("no text provided")

    # Decode input from base64
    try:
        embed = base64.b64decode(embed.encode('utf-8'))
        embed = np.frombuffer(embed, dtype=np.float32)
        text = base64.decodebytes(text.encode('utf-8')).decode('utf-8')
    except Exception as e:
        logging.log(logging.ERROR, e)
        return error_response("invalid speaker wav provided")

    # Apply seed
    if seed is None:
        seed = torch.seed()
    else:
        try:
            manual_seed = int(seed)
            torch.manual_seed(manual_seed)
            seed = manual_seed
        except Exception as e:
            logging.log(logging.ERROR, e)
            return error_response("invalid generation seed provided")
    logging.log(logging.INFO, "Using seed: %d" % seed)

    # Download synthesizer model from storage bucket
    if MODELS_BUCKET != "LOCAL" and not os.path.exists(SYNTHESIZER_MODEL_LOCAL_PATH):
        storage_client = storage.Client()
        encoders_bucket = storage_client.get_bucket(MODELS_BUCKET)
        encoder_model = encoders_bucket.blob(SYNTHESIZER_MODEL_BUCKET_PATH)
        encoder_model.download_to_filename(SYNTHESIZER_MODEL_LOCAL_PATH)

    # Load Speaker encoder
    if os.path.exists(SYNTHESIZER_MODEL_LOCAL_PATH):
        start = timer()
        synthesizer = Synthesizer(pathlib.Path(SYNTHESIZER_MODEL_LOCAL_PATH))
        logging.log(logging.INFO, "Successfully loaded synthesizer (%dms)." % int(1000 * (timer() - start)))
    else:
        return error_response("synthesizer model not found", 500)

    # Process multiline text as individual synthesis => Maybe Possibility for threaded speedup here
    texts = text.split("\n")
    embeds = [embed] * len(texts)
    sub_spectograms = synthesizer.synthesize_spectrograms(texts, embeds)
    full_spectogram = np.concatenate(sub_spectograms, axis=1)
    full_spectogram = full_spectogram.copy(order='C') # Make C-Contigous to allow encoding - might need to be reverted for vocoding

    # Build response
    response = {
        "synthesized": base64.b64encode(full_spectogram).decode('utf-8'),
        "synthesized_mel": render.spectogram(full_spectogram)
    }

    return flask.make_response(response)

# process_vocode_request
# Input params:
# - binary spectogram of synthesized text
# (- fixed neural network seed)
# Returns:
# - binary wav file of
def process_vocode_request(request_data):
    return {"success":"vocoder function triggered"}

# process_render_request -> Performs synthesize & vocode in a single step
# Input params:
# - speaker embedding
# - text to synthesize via embedding
# (- fixed neural network seed)
# Returns:
# - binary wav file of
def process_render_request(request_data):
    return {"success":"vocoder function triggered"}

# get_version returns basic info on this gcloud function
def get_version(request=None):
    response = {
        "function_name": "rtvc-gcloud-handler",
        "version": "0.1-beta",
        "usage": ""
    }
    if request != None:
        response["request_info"] = {
            "method": request.method,
            "args": request.args,
            "data": request.get_json(),
            "route": request.path
        }
    return response

# creates a flask response
def error_response(message, code=400):
    response = flask.make_response(message, code)
    return response

