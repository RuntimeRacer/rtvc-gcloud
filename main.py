import tracemalloc

import requests
import validators

import hashlib
import io
import os
import pathlib
import random
import tempfile
import ffmpeg

import const
import render
import flask
from flask_cors import CORS
import json
import base64
import logging
import torch
import numpy as np
import soundfile as sf

from time import perf_counter as timer

# SV2TTS
from config.hparams import sp
from encoder import inference as encoder
from encoder.audio import preprocess_wav
from synthesizer.models import base as syn_base
from synthesizer import inference as synthesizer
from vocoder import inference as vocoder, base as voc_base
from voicefixer import base as vf


# Cloud Run related stuff
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
@app.route("/render_batch", methods=['GET', 'POST'])
@app.route("/profile", methods=['GET', 'POST'])
def handle_request():
    # Evaluate request
    request = flask.request

    # Token Auth
    if check_token_auth(request.headers.get('Api-Key')) is False:
        # invalid client token provided
        return flask.make_response(const.ERROR_CLIENT_TOKEN_INVALID, 403)

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
    if 'render_batch' in request.path:  # Note: this one before the 'render-only' handler so we can properly handle
        response, code = process_render_batch_request(request_data)
        return flask.make_response(response, code)
    if 'render' in request.path:
        response, code = process_render_request(request_data)
        return flask.make_response(response, code)
    if 'profile' in request.path:
        response, code = process_profile_request(request_data)
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
    audio = request_data["speaker_audio"] if "speaker_audio" in request_data else None
    enhance_audio = True if "enhance_audio" in request_data and request_data["enhance_audio"] == 1 else False
    render_graph = request_data["render_graph"] if "render_graph" in request_data else False

    if audio is None:
        # no speaker audio provided
        return const.ERROR_NO_SAMPLE_PROVIDED, 400

    # Generate the spectogram - if this fails, audio data provided is invalid
    try:
        # Fetch audio data
        audio, success = fetch_data_from_url_or_decode(audio)
        if not success:
            return audio, 400

        # Save to temp file and convert to waveform using FFMPEG
        # Also create the file for the enhancement here, even though we might not need it.
        temp_audio = tempfile.NamedTemporaryFile(suffix='.audio', delete=False)
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', prefix=os.path.basename(temp_audio.name), delete=False)
        temp_wav_enh = tempfile.NamedTemporaryFile(suffix='.wav', prefix=os.path.basename(temp_audio.name).join('_enhanced'), delete=False)
        try:
            # Write Audio bytes to file and close it as well as the target file, so ffmpeg can write to it
            temp_audio.write(io.BytesIO(audio).getbuffer())
            temp_audio.close()
            temp_wav.close()
            temp_wav_enh.close()

            # Check if audio file is valid and not too long
            duration = float(ffmpeg.probe(temp_audio.name)["format"]["duration"])
            if duration > const.INPUT_MAX_LENGTH_SECONDS or duration < const.INPUT_MIN_LENGTH_SECONDS:
                # input audio needs to have a duration between 0.5 of 15 seconds
                return const.ERROR_SAMPLE_DURATION_INVALID, 400

            # Conversion using ffmpeg
            ffmpeg.input(temp_audio.name).output(temp_wav.name).run(overwrite_output=True)
            input_audio_file = temp_wav.name

            # Enhance Audio using voicefixer
            if enhance_audio:
                try:
                    # Load the model
                    if not load_voicefixer():
                        # voicefixer models not found
                        return const.ERROR_VOICEFIXER_NOT_FOUND, 500
                    # Enhance the audio
                    vf.restore(input=temp_wav.name, output=temp_wav_enh.name)
                    # Use enhanced file as encoder reference file
                    input_audio_file = temp_wav_enh.name
                except RuntimeError as e:
                    logging.log(logging.ERROR, e)
                    pass

            # Read in using BytesIO
            with open(input_audio_file, 'rb') as handle:
                wav = io.BytesIO(handle.read()).getvalue()
        except Exception as e:
            logging.log(logging.ERROR, e)
        finally:
            # Delete the temp files
            os.unlink(temp_audio.name)
            os.unlink(temp_wav.name)
            os.unlink(temp_wav_enh.name)

    except Exception as e:
        logging.log(logging.ERROR, e)
        # invalid speaker wav provided
        return const.ERROR_SAMPLE_DATA_INVALID, 400

    # Set Default Encoder Seed to 111
    torch.manual_seed(111)
    np.random.seed(111)
    os.environ["PYTHONHASHSEED"] = "111"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the model
    if not load_encoder():
        # encoder model not found
        return const.ERROR_ENCODER_NOT_FOUND, 500

    # process wav and generate embedding
    prep_wav = synthesizer.load_preprocess_wav(wav)
    encoder_wav = preprocess_wav(prep_wav)
    embed = encoder.embed_utterance(encoder_wav)

    if render_graph:
        # Generate the spectogram
        spectogram = synthesizer.make_spectrogram(wav)

    # Build response
    response = {
        "embed": base64.b64encode(embed).decode('utf-8'),
        "embed_graph": render.embedding(embed)  if render_graph else '',
        "embed_mel_graph": render.spectogram(spectogram) if render_graph else '',
        "version": const.BACKEND_VERSION
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
        # no speaker embedding provided
        return const.ERROR_NO_EMBEDDING_PROVIDED, 400
    if text is None or len(text) < 1:
        # no text provided
        return const.ERROR_NO_TEXT_PROVIDED, 400

    # Decode input from base64
    try:
        embed = base64.b64decode(embed.encode('utf-8'))
        embed = np.frombuffer(embed, dtype=np.float32)
        text = base64.decodebytes(text.encode('utf-8')).decode('utf-8')
    except Exception as e:
        logging.log(logging.ERROR, e)
        # invalid embedding or text provided
        return const.ERROR_EMBEDDING_OR_TEXT_INVALID, 400

    # Apply seed
    if seed is None:
        seed = random.randint(0, 4294967295)
    else:
        seed = int(seed)
    logging.log(logging.INFO, "Using seed: %d" % seed)

    # Ensure everything is properly set up
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the model
    if not load_synthesizer():
        # synthesizer model not found
        return const.ERROR_SYNTHESIZER_NOT_FOUND, 500

    # Perform the synthesis
    full_spectogram, breaks = do_synthesis(text, embed, speed_modifier, pitch_modifier, energy_modifier)

    # Build response
    breaks_json = json.dumps(breaks)
    full_spectogram_json = full_spectogram.copy(order='C')  # Make C-Contigous to allow encoding
    full_spectogram_json = json.dumps(full_spectogram_json.tolist())

    response = {
        "synthesized": {
            "mel": base64.b64encode(full_spectogram_json.encode('utf-8')).decode('utf-8'),
            "breaks": base64.b64encode(breaks_json.encode('utf-8')).decode('utf-8')
        },
        "synthesized_mel_graph": render.spectogram(full_spectogram),
        "version": const.BACKEND_VERSION
    }

    return response, 200

def do_synthesis(text, embed, speed_modifier, pitch_modifier, energy_modifier):
    # Process multiline text as individual synthesis => Maybe Possibility for threaded speedup here
    if synthesizer.get_model_type() == syn_base.MODEL_TYPE_TACOTRON:
        texts = text.split("\n")
    elif synthesizer.get_model_type() == syn_base.MODEL_TYPE_FORWARD_TACOTRON:
        texts = [text]
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
        # no synthesized data provided
        return const.ERROR_NO_SYNTHESIZED_DATA_PROVIDED, 400

    # Get mel and breaks
    syn_mel = synthesized["mel"] if "mel" in synthesized else None
    syn_breaks = synthesized["breaks"] if "breaks" in synthesized else None
    if syn_mel is None or syn_breaks is None:
        # no synthesized data provided
        return const.ERROR_NO_SYNTHESIZED_DATA_PROVIDED, 400

    # Decode input from base64
    try:
        syn_breaks = base64.b64decode(syn_breaks.encode('utf-8'))
        syn_breaks = json.loads(syn_breaks)
        syn_mel = base64.b64decode(syn_mel.encode('utf-8'))
        syn_mel = json.loads(syn_mel)
        syn_mel = np.array(syn_mel, dtype=np.float32)
    except Exception as e:
        logging.log(logging.ERROR, e)
        # invalid synthesis data provided
        return const.ERROR_SYNTHESIS_DATA_INVALID, 400

    # Apply seed
    if seed is None:
        seed = random.randint(0, 4294967295)
    else:
        seed = int(seed)
    logging.log(logging.INFO, "Using seed: %d" % seed)

    # Ensure everything is properly set up
    vocoder.set_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the model
    if not load_vocoder():
        # vocoder model not found
        return const.ERROR_VOCODER_NOT_FOUND, 500

    # Perform the vocoding
    wav_string = do_vocode(syn_mel, syn_breaks)

    # Build response
    response = {
        "generated_wav": base64.b64encode(wav_string).decode('utf-8'),
        "version": const.BACKEND_VERSION
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
    #wav = preprocess_wav(wav)  # Trim silences
    #wav = nr.reduce_noise(y=wav, sr=sp.sample_rate, stationary=False, n_fft=sp.n_fft, hop_length=sp.hop_size,
    #                      win_length=sp.win_size, n_jobs=-1)
    wav = wav / np.abs(wav).max() * 0.97  # Normalize

    # Encode as WAV
    with io.BytesIO() as handle:
        sf.write(handle, wav, samplerate=sp.sample_rate, format='wav')
        wav_string = handle.getvalue()
    return wav_string

# process_render_request -> Performs synthesize & vocode in a single step
# Input params:
# - speaker embedding
# - speed modifier for generation
# - pitch modifier for generation
# - energy modifier for generation
# - text to synthesize via embedding
# (- fixed neural network seed)
# (- flag whether to render a spectogram)
# (- video image data; creates a video instead of audio with the provided image(s) in the background)
# Returns:
# - binary wav file of results
def process_render_request(request_data):
    # Gather data from request
    embed = request_data["speaker_embed"] if "speaker_embed" in request_data else None
    speed_modifier = request_data["speed_modifier"] if "speed_modifier" in request_data else None
    pitch_modifier = request_data["pitch_modifier"] if "pitch_modifier" in request_data else None
    energy_modifier = request_data["energy_modifier"] if "energy_modifier" in request_data else None
    text = request_data["text"] if "text" in request_data else None
    seed = request_data["seed"] if "seed" in request_data else None
    render_graph = request_data["render_graph"] if "render_graph" in request_data else False
    video_image_data = request_data["video_image_data"] if "video_image_data" in request_data else None

    # Check input
    if embed is None:
        # no speaker embedding provided
        return const.ERROR_NO_EMBEDDING_PROVIDED, 400
    if text is None or len(text) < 1:
        # no text provided
        return const.ERROR_NO_TEXT_PROVIDED, 400

    # Decode input from base64
    try:
        embed = base64.b64decode(embed.encode('utf-8'))
        embed = np.frombuffer(embed, dtype=np.float32)
        text = base64.decodebytes(text.encode('utf-8')).decode('utf-8')
    except Exception as e:
        logging.log(logging.ERROR, e)
        # invalid embedding or text provided
        return const.ERROR_EMBEDDING_OR_TEXT_INVALID, 400

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
            # invalid generation seed provided
            return const.ERROR_SEED_INVALID, 400
    logging.log(logging.INFO, "Using seed: %d" % seed)

    # Load the models
    if not load_synthesizer():
        # synthesizer model not found
        return const.ERROR_SYNTHESIZER_NOT_FOUND, 500
    if not load_vocoder():
        # vocoder model not found
        return const.ERROR_VOCODER_NOT_FOUND, 500

    # Perform the synthesis
    syn_mel, syn_breaks = do_synthesis(text, embed, speed_modifier, pitch_modifier, energy_modifier)

    # Perform the vocoding
    wav_string = do_vocode(syn_mel, syn_breaks)

    if render_graph:
        spectogram = synthesizer.make_spectrogram(wav_string)

    rendered_video = ''
    if video_image_data:
        # Render a video using provided background image or video and generated audio (request from a friend)
        # if this failed, image data provided is invalid
        try:
            # Fetch video data
            video_image_data, success = fetch_data_from_url_or_decode(video_image_data)
            if not success:
                return video_image_data, 400

            # Save to temp file
            temp_image = tempfile.NamedTemporaryFile(suffix='.image', delete=False)
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', prefix=os.path.basename(temp_image.name), delete=False)
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', prefix=os.path.basename(temp_image.name), delete=False)
            try:
                # Write image and wav bytes to file and close it as well as the target file, so ffmpeg can write to it
                temp_image.write(io.BytesIO(video_image_data).getbuffer())
                temp_audio.write(io.BytesIO(wav_string).getbuffer())
                temp_image.close()
                temp_audio.close()
                temp_video.close()

                # Video creation using ffmpeg
                image_input = ffmpeg.input(temp_image.name)
                audio_input = ffmpeg.input(temp_audio.name)
                video_render = ffmpeg.output(
                    image_input,
                    audio_input,
                    filename=temp_video.name,
                    loop=1
                )
                video_render.run(overwrite_output=True)

                # Read in using BytesIO
                with open(temp_video.name, 'rb') as handle:
                    rendered_video = io.BytesIO(handle.read()).getvalue()
            finally:
                # Delete the temp files
                os.unlink(temp_image.name)
                os.unlink(temp_audio.name)
                os.unlink(temp_video.name)

        except Exception as e:
            logging.log(logging.ERROR, e)
            # invalid image provided
            return const.ERROR_IMAGE_DATA_INVALID, 400

    # Build response
    response = {
        "generated_wav": base64.b64encode(wav_string).decode('utf-8'),
        "rendered_mel_graph": render.spectogram(spectogram) if render_graph else '',
        "rendered_video": base64.b64encode(rendered_video).decode('utf-8') if len(rendered_video) > 0 else '',
        "version": const.BACKEND_VERSION
    }

    return response, 200


# process_render_batch_request -> Performs synthesize & vocode in a single step for multiple lines
# Input params:
# - speaker embedding
# - speed modifier for generation
# - pitch modifier for generation
# - energy modifier for generation
# - [texts to synthesize via embedding]
# (- fixed neural network seed)
# (- flag whether to render a spectogram for each wav)
# Returns:
# - [binary wav files for provide texts, in identical order]
def process_render_batch_request(request_data):
    # Gather data from request
    embed = request_data["speaker_embed"] if "speaker_embed" in request_data else None
    speed_modifier = request_data["speed_modifier"] if "speed_modifier" in request_data else None
    pitch_modifier = request_data["pitch_modifier"] if "pitch_modifier" in request_data else None
    energy_modifier = request_data["energy_modifier"] if "energy_modifier" in request_data else None
    texts = request_data["texts"] if "texts" in request_data else None
    seed = request_data["seed"] if "seed" in request_data else None
    render_graph = request_data["render_graph"] if "render_graph" in request_data else False

    # Check input
    if embed is None:
        # no speaker embedding provided
        return const.ERROR_NO_EMBEDDING_PROVIDED, 400
    if texts is None or not isinstance(texts, list):
        # text data needs to be a list
        return const.ERROR_TEXTS_NOT_A_LIST, 400
    if len(texts) > 10:  # FIXME: make this an env var
        # maximum batching amount is 10
        return const.ERROR_BATCHING_AMOUNT_EXCEEDED, 400
    for idx, text in enumerate(texts):
        if len(text) < 1:
            # one of the texts is empty
            return const.ERROR_TEXTS_EMPTY, 400

    # Decode input from base64
    try:
        embed = base64.b64decode(embed.encode('utf-8'))
        embed = np.frombuffer(embed, dtype=np.float32)
        text_data = []
        for text in texts:
            text_data.append(base64.decodebytes(text.encode('utf-8')).decode('utf-8'))
    except Exception as e:
        logging.log(logging.ERROR, e)
        # invalid embedding or text data provided
        return const.ERROR_EMBEDDING_OR_TEXT_INVALID, 400

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
            # invalid generation seed provided
            return const.ERROR_SEED_INVALID, 400
    logging.log(logging.INFO, "Using seed: %d" % seed)

    # Load the models
    if not load_synthesizer():
        # synthesizer model not found
        return const.ERROR_SYNTHESIZER_NOT_FOUND, 500
    if not load_vocoder():
        # vocoder model not found
        return const.ERROR_VOCODER_NOT_FOUND, 500

    # Perform the batch render
    wav_strings = []
    spectograms = []
    for text in text_data:
        # Perform the synthesis
        syn_mel, syn_breaks = do_synthesis(text, embed, speed_modifier, pitch_modifier, energy_modifier)
        # Perform the vocoding
        wav_string = do_vocode(syn_mel, syn_breaks)

        if render_graph:
            spectogram = synthesizer.make_spectrogram(wav_string)
            spectogram = render.spectogram(spectogram)
            spectograms.append(spectogram)

        # Encode the wav & append to list
        wav_string = base64.b64encode(wav_string).decode('utf-8')
        wav_strings.append(wav_string)

    # Build response
    response = {
        "generated_wavs": wav_strings,
        "rendered_mel_graphs": spectograms if render_graph else [],
        "version": const.BACKEND_VERSION
    }

    return response, 200


def process_profile_request(request_data):
    if os.environ.get("PROFILE_MEMORY") == "":
        # profiling is currently disabled
        return const.ERROR_PROFILING_DISABLED, 400

    # Get Memory Snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    # Build response
    stats_list = [str(stat) for stat in top_stats]
    response = json.dumps(stats_list)

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
    else:
        return True


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


def load_voicefixer():
    if not vf.is_loaded():
        if os.path.exists(const.VOICEFIXER_ANALYZER_PATH) and os.path.exists(const.VOICEFIXER_VOCODER_PATH):
            start = timer()
            vf.load_model(const.VOICEFIXER_ANALYZER_PATH, const.VOICEFIXER_VOCODER_PATH)
            logging.log(logging.INFO, "Successfully loaded voicefixer (%dms)." % int(1000 * (timer() - start)))
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


def fetch_data_from_url_or_decode(url_or_data, decode_data=True):
    # Check if data is provided in binary format or is an URL
    if validators.url(url_or_data):
        # Check if the URL is allowed to download from
        valid_target = False
        whitelist_urls = os.environ.get("WHITELIST_URLS")
        whitelist_urls = whitelist_urls.split(",")
        for url in whitelist_urls:
            if url in url_or_data:
                valid_target = True
                break

        if not valid_target:
            # invalid target URL provided
            return const.ERROR_DATA_URL_INVALID, False

        # Download file and store in buffer
        r = requests.get(url_or_data)
        if r.status_code != 200:
            # target url did not return HTTP-200
            return const.ERROR_SAMPLE_URL_RESPONSE, False

        # Return data fetched from URL
        return r.content, True

    if decode_data:
        # Decode the image from payload
        return base64.b64decode(url_or_data), True

    return url_or_data, True


# preload_models loads all models into memory on app startup (if flag is set)
def preload_models():
    load_encoder()
    load_synthesizer()
    load_vocoder()
    load_voicefixer()


# Preload Models if flag is set
if os.environ.get("PRELOAD") != "":
    preload_models()

# Init Main process
if __name__ == "__main__":
    if os.environ.get("PROFILE_MEMORY") != "":
        tracemalloc.start()

    # Run the webserver for handling requests - see also:
    # https://stackoverflow.com/questions/51025893/flask-at-first-run-do-not-use-the-development-server-in-a-production-environmen
    from waitress import serve
    # app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
