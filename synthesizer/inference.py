import hashlib
import io
from pathlib import Path
from typing import List, Union

import librosa
import numpy as np
import torch

from config.hparams import preprocessing, sp
from config.hparams import tacotron as hp_tacotron
from synthesizer import audio
from synthesizer.models import base
from synthesizer.utils.text import text_to_sequence


class Synthesizer:
    
    def __init__(self, model_fpath: Path, verbose=True):
        """
        The model isn't instantiated and loaded in memory until needed or until load() is called.
        
        :param model_fpath: path to the trained model file
        :param verbose: if False, prints less information when using the model
        """
        self.model_fpath = model_fpath
        self.verbose = verbose
 
        # Check for GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.verbose:
            print("Synthesizer using device:", self.device)
        
        # Synthesizer model and type will be instantiated later on first use.
        self._model = None
        self._model_type = None

    def is_loaded(self):
        """
        Whether the model is loaded in memory.
        """
        return self._model is not None

    def get_model_type(self):
        """
        Get the model type
        """
        # Load the model on the first request.
        if not self.is_loaded():
            self.load()
        return self._model_type

    def load(self):
        """
        Instantiates and loads the model given the weights file that was passed in the constructor.
        Model type is determined from weights file, however if type is not included it will assume default Tacotron.
        """

        # Load weights
        checkpoint = torch.load(str(self.model_fpath), map_location=self.device)
        self._model_type = base.MODEL_TYPE_TACOTRON
        if "model_type" in checkpoint:
            self._model_type = checkpoint["model_type"]

        # Build model based on detected model type
        try:
            self._model = base.init_syn_model(self._model_type, self.device)
        except NotImplementedError as e:
            print(str(e))
            return

        # Load checkpoint data into model
        self._model.load(self.model_fpath, optimizer=None, checkpoint=checkpoint)
        self._model.eval()

        if self.verbose:
            print("Loaded synthesizer of model '%s' at path '%s'." % (self._model_type, self.model_fpath.name))
            print("Model has been trained to step %d." % (self._model.state_dict()["step"]))

    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False,
                                speed_modifier=1.0,
                                pitch_function=None,
                                energy_function=None):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.

        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256) 
        :param return_alignments: if True, a matrix representing the alignments between the 
        characters
        and each decoder output step will be returned for each spectrogram
        :param speed_modifier: For advanced models; modifies speed of speech.
        :param pitch_function: For advanced models; modifies pitch of speech.
        :param energy_function: For advanced models; modifies energy of speech.
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the
        sequence length of spectrogram i, and possibly the alignments.
        """
        # Load the model on the first request.
        if not self.is_loaded():
            self.load()

        # Preprocess text inputs
        inputs = [text_to_sequence(text.strip(), preprocessing.cleaner_names) for text in texts]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        # Batch inputs
        batched_inputs = [inputs[i:i + preprocessing.synthesis_batch_size]
                          for i in range(0, len(inputs), preprocessing.synthesis_batch_size)]
        batched_embeds = [embeddings[i:i + preprocessing.synthesis_batch_size]
                          for i in range(0, len(embeddings), preprocessing.synthesis_batch_size)]

        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            if self.verbose:
                print(f"\n| Generating {i}/{len(batched_inputs)}")

            # Pad texts so they are all the same length
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Stack speaker embeddings into 2D array for batch processing
            speaker_embeds = np.stack(batched_embeds[i-1])

            # Convert to tensor
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

            # Inference
            if self._model_type == base.MODEL_TYPE_TACOTRON:
                _, mels, alignments = self._model.generate(chars, speaker_embeddings)
                mels = mels.detach().cpu().numpy()
                for m in mels:
                    # Trim silence from end of each spectrogram
                    while np.max(m[:, -1]) < hp_tacotron.stop_threshold:
                        m = m[:, :-1]
                    specs.append(m)
            elif self._model_type == base.MODEL_TYPE_FORWARD_TACOTRON:
                _, mels, _, _, _ = self._model.generate(chars, speaker_embeddings, speed_modifier, pitch_function, energy_function)
                mels = mels.detach().cpu().numpy()
                for m in mels:
                    specs.append(m)

        if self.verbose:
            print("\n\nDone.\n")
        return (specs, alignments) if return_alignments else specs


_model = None # type: Synthesizer

def load_model(weights_fpath, verbose=True):
    global _model, _device, _model_type

    if torch.cuda.is_available():
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')

    # Load model weights from provided model path
    _model = Synthesizer(weights_fpath, verbose)
    _model.load()

def is_loaded():
    return _model is not None and _model.is_loaded()

def get_model_type():
    if not is_loaded():
        raise Exception("Please load Synthesizer in memory before using it")
    else:
        return _model.get_model_type()

def synthesize_spectrograms(texts: List[str], embeddings: Union[np.ndarray, List[np.ndarray]], return_alignments=False,
                            speed_modifier=1.0, pitch_function=None, energy_function=None):
    if not is_loaded():
        raise Exception("Please load Synthesizer in memory before using it")
    return _model.synthesize_spectrograms(texts=texts, embeddings=embeddings, return_alignments=return_alignments,
                                          speed_modifier=speed_modifier, pitch_function=pitch_function,
                                          energy_function=energy_function)

def load_preprocess_wav(wav):
    """
    Loads and preprocesses an audio file under the same conditions the audio files were used to
    train the synthesizer.
    """
    wav = librosa.load(io.BytesIO(wav), sr=sp.sample_rate)[0]
    wav_md5 = hashlib.md5(wav)
    print("MD5 Checks - Wav: {0}".format(wav_md5.hexdigest()))

    if preprocessing.rescale:
        wav = wav / np.abs(wav).max() * preprocessing.rescaling_max

    prep_wav_md5 = hashlib.md5(wav)
    print("MD5 Checks - Preprocessed Wav: {0}".format(prep_wav_md5.hexdigest()))
    return wav

def make_spectrogram(wav):
    """
    Creates a mel spectrogram from an audio file in the same manner as the mel spectrograms that
    were fed to the synthesizer when training.
    """
    wav = load_preprocess_wav(wav)

    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    return mel_spectrogram

def griffin_lim(mel):
    """
    Inverts a mel spectrogram using Griffin-Lim. The mel spectrogram is expected to have been built
    with the same parameters present in hparams.py.
    """
    return audio.inv_mel_spectrogram(mel)

def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)
