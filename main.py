import io

import flask
import base64
import logging
from time import perf_counter as timer
from google.cloud import storage

# SV2TTS
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# Cloud Function related stuff
app = flask.Flask(__name__)

MODELS_BUCKET = 'kajispeech-models'
ENCODER_MODEL_PATH = '/tmp/encoder.pt'

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
    wav = request_data["speaker_wav"]

    # Generate the spectogram - if this fails, audio data provided is invalid
    try:
        # Decode the wav from payload
        wav = base64.b64decode(wav)
        # Generate the spectogram
        spectogram = Synthesizer.make_spectrogram(wav)
    except:
        return error_response("invalid speaker wav provided")

    # Download speaker encoder model from storage bucket
    storage_client = storage.Client()
    encoders_bucket = storage_client.get_bucket(MODELS_BUCKET)
    encoder_model = encoders_bucket.blob("encoders/encoder_new_768.pt")
    encoder_model.download_to_filename(ENCODER_MODEL_PATH)

    # Load Speaker encoder
    start = timer()
    encoder.load_model(ENCODER_MODEL_PATH)
    logging.log(logging.DEBUG, "Successfully loaded encoder (%dms)." % int(1000 * (timer() - start)))

    # process wav and generate embedding
    encoder_wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(encoder_wav)

    # Generate the embed graph
    # TODO

    # Build response
    response = {
        "embed": base64.b64encode(embed),
        "embed_graph": None,
        "embed_mel": base64.b64encode(spectogram)
    }

    return flask.make_response(response)

# process_synthesize_request
# Input params:
# - speaker embedding
# - text to synthesize via embedding
# (- fixed neural network seed)
# Returns:
# - binary spectogram of synthesized text
def process_synthesize_request(request_data):
    # TODO
    return {"success":"synthesizer function triggered"}

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