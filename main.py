import io
import os
import pathlib
import const
import render
import flask
from flask_cors import CORS
import json
import base64
import logging
import torch
import numpy as np
import noisereduce as nr
import soundfile as sf

from time import perf_counter as timer

# SV2TTS
from config.hparams import sp
from encoder import inference as encoder
from encoder.audio import preprocess_wav
from synthesizer.models import base as syn_base
from synthesizer import inference as synthesizer
from vocoder.models import base as voc_base
from vocoder import inference as vocoder

# Cloud Function related stuff
app = flask.Flask(__name__)
CORS(app)


# entry point of this gcloud function
# Expects requests to be sent via POST method.
# In case of a non-POST Request it will respond with get_version().
@app.route("/", methods=['GET', 'POST'])
@app.route("/encode", methods=['GET', 'POST'])
@app.route("/synthesize", methods=['GET', 'POST'])
@app.route("/vocode", methods=['GET', 'POST'])
@app.route("/render", methods=['GET', 'POST'])
def handle_request():
    # Evaluate request
    request = flask.request

    # Token Auth
    if check_token_auth(request.headers.get('api-key')) is False:
        response = {
            "error": "invalid client token provided"
        }
        return flask.make_response(response, 403)

    # Default route for non-post or bad request
    method = request.method
    if method != 'POST' or not request.is_json:
        response, code = get_version(request)
        return flask.make_response(response, code)

    # Parse request data
    request_data = request.get_json()

    # Get route and forward request
    if 'encode' in request.path:
        response, code = process_encode_request(request_data)
        return flask.make_response(response, code)
    if 'synthesize' in request.path:
        response, code = process_synthesize_request(request_data)
        return flask.make_response(response, code)
    if 'vocode' in request.path:
        response, code = process_vocode_request(request_data)
        return flask.make_response(response, code)
    if 'render' in request.path:
        response, code = process_render_request(request_data)
        return flask.make_response(response, code)
    else:
        response, code = get_version(request)
        return flask.make_response(response, code)

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
        spectogram = synthesizer.make_spectrogram(wav)
    except Exception as e:
        logging.log(logging.ERROR, e)
        return "invalid speaker wav provided", 400

    # Load the model
    if not load_encoder():
        return "encoder model not found", 500

    # process wav and generate embedding
    encoder_wav = preprocess_wav(wav)
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
    speed_modifier = request_data["speed_modifier"] if "speed_modifier" in request_data else None
    pitch_modifier = request_data["pitch_modifier"] if "pitch_modifier" in request_data else None
    energy_modifier = request_data["energy_modifier"] if "energy_modifier" in request_data else None

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

    # Load the model
    if not load_synthesizer():
        return "synthesizer model not found", 500

    # Perform the synthesis
    full_spectogram, breaks = do_synthesis(text, embed, speed_modifier, pitch_modifier, energy_modifier)

    # Build response
    breaks_json = json.dumps(breaks)
    full_spectogram_json = full_spectogram.copy(order='C')  # Make C-Contigous to allow encoding - might need to be reverted for vocoding
    full_spectogram_json = json.dumps(full_spectogram_json.tolist())

    response = {
        "synthesized": {
            "mel": base64.b64encode(full_spectogram_json.encode('utf-8')).decode('utf-8'),
            "breaks": base64.b64encode(breaks_json.encode('utf-8')).decode('utf-8')
        },
        "synthesized_mel_graph": render.spectogram(full_spectogram)
    }

    return response, 200

def do_synthesis(text, embed, speed_modifier, pitch_modifier, energy_modifier):
    # Process multiline text as individual synthesis => Maybe Possibility for threaded speedup here
    texts = text.split("\n")
    embeds = [embed] * len(texts)

    # Params for advanced model
    speed_function = 1.0
    pitch_function = lambda x: x
    energy_function = lambda x: x
    if synthesizer.get_model_type() == syn_base.MODEL_TYPE_FORWARD_TACOTRON:
        speed_function = float(speed_modifier)
        pitch_function = lambda x: x * float(pitch_modifier)
        energy_function = lambda x: x * float(energy_modifier)

    sub_spectograms = synthesizer.synthesize_spectrograms(texts=texts, embeddings=embeds, speed_modifier=speed_function,
                                                          pitch_function=pitch_function,
                                                          energy_function=energy_function)

    # Get speech breaks and store as JSON list
    breaks = [subspec.shape[1] for subspec in sub_spectograms]

    # Combine full spectogram
    full_spectogram = np.concatenate(sub_spectograms, axis=1)

    return full_spectogram, breaks

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
    if not load_vocoder():
        return "vocoder model not found", 500

    # Perform the vocoding
    wav_string = do_vocode(syn_mel, syn_breaks)

    # Build response
    response = {
        "generated_wav": base64.b64encode(wav_string).decode('utf-8')
    }

    return response, 200

def do_vocode(syn_mel, syn_breaks):
    # Apply vocoder on mel
    wav = vocoder.infer_waveform(syn_mel)

    # Add breaks
    b_ends = np.cumsum(np.array(syn_breaks) * sp.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
    syn_breaks = [np.zeros(int(0.15 * sp.sample_rate))] * len(syn_breaks)
    wav = np.concatenate([i for w, b in zip(wavs, syn_breaks) for i in (w, b)])

    # Apply optimizations
    wav = preprocess_wav(wav)  # Trim silences
    #wav = nr.reduce_noise(y=wav, sr=sp.sample_rate, stationary=False, n_fft=sp.n_fft, hop_length=sp.hop_size,
    #                      win_length=sp.win_size, n_jobs=-1)
    wav = wav / np.abs(wav).max() * 0.97  # Normalize

    # Encode as WAV
    with io.BytesIO() as handle:
        sf.write(handle, wav.astype(np.float32), samplerate=sp.sample_rate, format='wav')
        wav_string = handle.getvalue()
    return wav_string

# process_render_request -> Performs synthesize & vocode in a single step
# Input params:
# - speaker embedding
# - text to synthesize via embedding
# (- fixed neural network seed)
# Returns:
# - binary wav file of
def process_render_request(request_data):
    # Gather data from request
    embed = request_data["speaker_embed"] if "speaker_embed" in request_data else None
    text = request_data["text"] if "text" in request_data else None
    seed = request_data["seed"] if "seed" in request_data else None
    speed_modifier = request_data["speed_modifier"] if "speed_modifier" in request_data else None
    pitch_modifier = request_data["pitch_modifier"] if "pitch_modifier" in request_data else None
    energy_modifier = request_data["energy_modifier"] if "energy_modifier" in request_data else None

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

    # Load the models
    if not load_synthesizer():
        return "synthesizer model not found", 500
    if not load_vocoder():
        return "vocoder model not found", 500

    # Perform the synthesis
    syn_mel, syn_breaks = do_synthesis(text, embed, speed_modifier, pitch_modifier, energy_modifier)

    # Perform the vocoding
    wav_string = do_vocode(syn_mel, syn_breaks)

    # Build response
    response = {
        "generated_wav": base64.b64encode(wav_string).decode('utf-8')
    }

    return response, 200

# load_encoder loads the encoder into memory
def load_encoder():
    if not encoder.is_loaded():
        # Load Speaker encoder
        if os.path.exists(const.ENCODER_PATH):
            start = timer()
            encoder.load_model(pathlib.Path(const.ENCODER_PATH))
            logging.log(logging.INFO, "Successfully loaded encoder (%dms)." % int(1000 * (timer() - start)))
            return True
        else:
            return False

# load_synthesizer loads the synthesizer into memory
def load_synthesizer():
    if not synthesizer.is_loaded():
        if os.path.exists(const.SYNTHESIZER_PATH):
            start = timer()
            synthesizer.load_model(pathlib.Path(const.SYNTHESIZER_PATH))
            logging.log(logging.INFO, "Successfully loaded synthesizer (%dms)." % int(1000 * (timer() - start)))
            return True
        else:
            return False
    else:
        return True

# load_vocoder loads the vocoder into memory
def load_vocoder():
    if not vocoder.is_loaded():
        # Determine vocoder type
        vocoder_type = os.environ.get("VOC_TYPE")
        if vocoder_type == voc_base.VOC_TYPE_CPP:
            vocoder_path = const.VOCODER_BINARY_PATH
        else:
            vocoder_path = const.VOCODER_PATH

        # Load vocoder
        if os.path.exists(vocoder_path):
            start = timer()
            vocoder.load_model(weights_fpath=pathlib.Path(vocoder_path), voc_type=vocoder_type)
            logging.log(logging.INFO, "Successfully loaded vocoder (%dms)." % int(1000 * (timer() - start)))
            return True
        else:
            return False
    else:
        return True



# check_token_auth validates a provided endpoint token
def check_token_auth(client_token):
    env_token = os.environ.get("END_POINT_TOKEN", "")
    if len(env_token) == 0 or client_token != env_token:
        return False
    return True

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
            "data": request.get_json() if request.is_json else "",
            "route": request.path
        }
    return response, 200

# preload_models loads all models into memory on app startup (if flag is set)
def preload_models():
    load_encoder()
    load_synthesizer()
    load_vocoder()

if __name__ == "__main__":
    # Preload Models if flag is set
    if os.environ.get("PRELOAD"):
        preload_models()

    # Run the webserver for handling requests
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))