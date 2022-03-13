import io
import os
import pathlib

import render
import flask
import json
import base64
import logging
import torch
import numpy as np
import soundfile as sf

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

# entry point of this gcloud function
# Expects requests to be sent via POST method.
# In case of a non-POST Request it will respond with get_version().
@app.route("/")
@app.route("/encode")
@app.route("/synthesize")
@app.route("/vocode")
@app.route("/render")
def handle_request(request: flask.Request):
    # Add CORS header to each response
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    # Parse request data anyways
    method = request.method
    request_data = request.get_json()
    if method != 'POST' or not request_data:
        response, code = get_version(request)
        return flask.make_response(response, code, headers)

    # Get route and forward request
    if 'encode' in request.path:
        response, code = process_encode_request(request_data)
        return flask.make_response(response, code, headers)
    if 'synthesize' in request.path:
        response, code = process_synthesize_request(request_data)
        return flask.make_response(response, code, headers)
    if 'vocode' in request.path:
        response, code = process_vocode_request(request_data)
        return flask.make_response(response, code, headers)
    if 'render' in request.path:
        response, code = process_render_request(request_data)
        return flask.make_response(response, code, headers)
    else:
        response, code = get_version(request)
        return flask.make_response(response, code, headers)

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
        return "no speaker wav provided", 400

    # Generate the spectogram - if this fails, audio data provided is invalid
    try:
        # Decode the wav from payload
        wav = base64.b64decode(wav)
        # Generate the spectogram
        spectogram = Synthesizer.make_spectrogram(wav)
    except Exception as e:
        logging.log(logging.ERROR, e)
        return "invalid speaker wav provided", 400

    # Load the model
    if not encoder.is_loaded():
        # Download speaker encoder model from storage bucket
        if MODELS_BUCKET != "LOCAL" and not os.path.exists(ENCODER_MODEL_LOCAL_PATH):
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(MODELS_BUCKET)
            model = bucket.blob(ENCODER_MODEL_BUCKET_PATH)
            model.download_to_filename(ENCODER_MODEL_LOCAL_PATH)

        # Load Speaker encoder
        if os.path.exists(ENCODER_MODEL_LOCAL_PATH):
            start = timer()
            encoder.load_model(pathlib.Path(ENCODER_MODEL_LOCAL_PATH))
            logging.log(logging.INFO, "Successfully loaded encoder (%dms)." % int(1000 * (timer() - start)))
        else:
            return "encoder model not found", 500

    # process wav and generate embedding
    encoder_wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(encoder_wav)

    # Build response
    response = {
        "embed": base64.b64encode(embed).decode('utf-8'),
        "embed_graph": render.embedding(embed),
        "embed_mel_graph": render.spectogram(spectogram)
    }

    return response, 200

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
        return "no speaker embedding provided", 400
    if text is None or len(text) < 1:
        return "no text provided", 400

    # Decode input from base64
    try:
        embed = base64.b64decode(embed.encode('utf-8'))
        embed = np.frombuffer(embed, dtype=np.float32)
        text = base64.decodebytes(text.encode('utf-8')).decode('utf-8')
    except Exception as e:
        logging.log(logging.ERROR, e)
        return "invalid embedding or text provided", 400

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
            return "invalid generation seed provided", 400
    logging.log(logging.INFO, "Using seed: %d" % seed)

    # Load the model FIXME: Make this static method as the other 2
    # Download synthesizer model from storage bucket
    if MODELS_BUCKET != "LOCAL" and not os.path.exists(SYNTHESIZER_MODEL_LOCAL_PATH):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(MODELS_BUCKET)
        model = bucket.blob(SYNTHESIZER_MODEL_BUCKET_PATH)
        model.download_to_filename(SYNTHESIZER_MODEL_LOCAL_PATH)

    # Load synthesizer
    if os.path.exists(SYNTHESIZER_MODEL_LOCAL_PATH):
        start = timer()
        synthesizer = Synthesizer(pathlib.Path(SYNTHESIZER_MODEL_LOCAL_PATH))
        logging.log(logging.INFO, "Successfully loaded synthesizer (%dms)." % int(1000 * (timer() - start)))
    else:
        return "synthesizer model not found", 500

    # Process multiline text as individual synthesis => Maybe Possibility for threaded speedup here
    texts = text.split("\n")
    embeds = [embed] * len(texts)
    sub_spectograms = synthesizer.synthesize_spectrograms(texts, embeds)

    # Get speech breaks and store as JSON list
    breaks = [subspec.shape[1] for subspec in sub_spectograms]
    breaks_json = json.dumps(breaks)

    # Combine full spectogram
    full_spectogram = np.concatenate(sub_spectograms, axis=1)
    full_spectogram = full_spectogram.copy(order='C') # Make C-Contigous to allow encoding - might need to be reverted for vocoding
    full_spectogram_json = json.dumps(full_spectogram.tolist())

    # Build response
    response = {
        "synthesized": {
            "mel": base64.b64encode(full_spectogram_json.encode('utf-8')).decode('utf-8'),
            "breaks": base64.b64encode(breaks_json.encode('utf-8')).decode('utf-8')
        },
        "synthesized_mel_graph": render.spectogram(full_spectogram)
    }

    return response, 200

# process_vocode_request
# Input params:
# - binary spectogram of synthesized text
# (- fixed neural network seed)
# Returns:
# - binary wav file of vocoded spectogram
def process_vocode_request(request_data):
    # Gather data from request
    synthesized = request_data["synthesized"] if "synthesized" in request_data else None
    seed = request_data["seed"] if "seed" in request_data else None

    # Check input
    if synthesized is None:
        return "no synthesized data provided", 400

    # Get mel and breaks
    syn_mel = synthesized["mel"] if "mel" in synthesized else None
    syn_breaks = synthesized["breaks"] if "breaks" in synthesized else None
    if syn_mel is None or syn_breaks is None:
        return "invalid synthesis data provided", 400

    # Decode input from base64
    try:
        syn_breaks = base64.b64decode(syn_breaks.encode('utf-8'))
        syn_breaks = json.loads(syn_breaks)
        syn_mel = base64.b64decode(syn_mel.encode('utf-8'))
        syn_mel = json.loads(syn_mel)
        syn_mel = np.array(syn_mel, dtype=np.float32)
    except Exception as e:
        logging.log(logging.ERROR, e)
        return "invalid synthesis data provided", 400

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
            return "invalid generation seed provided", 400
    logging.log(logging.INFO, "Using seed: %d" % seed)

    # Load the model
    if not vocoder.is_loaded():
        # Download vocoder model from storage bucket
        if MODELS_BUCKET != "LOCAL" and not os.path.exists(VOCODER_MODEL_LOCAL_PATH):
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(MODELS_BUCKET)
            model = bucket.blob(VOCODER_MODEL_BUCKET_PATH)
            model.download_to_filename(VOCODER_MODEL_LOCAL_PATH)

        # Load vocoder
        if os.path.exists(VOCODER_MODEL_LOCAL_PATH):
            start = timer()
            vocoder.load_model(pathlib.Path(VOCODER_MODEL_LOCAL_PATH))
            logging.log(logging.INFO, "Successfully loaded vocoder (%dms)." % int(1000 * (timer() - start)))
        else:
            return "vocoder model not found", 500

    # Apply vocoder on mel
    wav = vocoder.infer_waveform(syn_mel)

    # Add breaks
    b_ends = np.cumsum(np.array(syn_breaks) * Synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
    syn_breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(syn_breaks)
    wav = np.concatenate([i for w, b in zip(wavs, syn_breaks) for i in (w, b)])

    # Apply optimizations
    wav = encoder.preprocess_wav(wav) # Trim silences
    wav = wav / np.abs(wav).max() * 0.97 # Normalize

    # Encode as WAV
    with io.BytesIO() as handle:
        sf.write(handle, wav.astype(np.float32), samplerate=Synthesizer.sample_rate, format='wav')
        wav_string = handle.getvalue()

    # Build response
    response = {
        "generated_wav": base64.b64encode(wav_string).decode('utf-8')
    }

    return response, 200

# process_render_request -> Performs synthesize & vocode in a single step
# Input params:
# - speaker embedding
# - text to synthesize via embedding
# (- fixed neural network seed)
# Returns:
# - binary wav file of
def process_render_request(request_data):
    return {"success":"vocoder function triggered"}, 200

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
    return response, 200
