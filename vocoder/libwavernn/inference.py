import math

from concurrent.futures import ThreadPoolExecutor

from config.hparams import sp, wavernn_fatchord, wavernn_geneing, wavernn_runtimeracer
from vocoder.models import base
from vocoder.audio import decode_mu_law, de_emphasis
from pathlib import Path
import numpy as np
import psutil
import WaveRNNVocoder


class Vocoder:

    def __init__(
            self,
            model_fpath: Path,
            model_type: str,
            verbose=True):
        """
        The model isn't instantiated and loaded in memory until needed or until load() is called.
        :param model_fpath: path to the trained, binary converted model file
        :param model_type: type of the model; for proper hparam mapping
        :param verbose: if False, prints less information when using the model
        """
        self.model_fpath = model_fpath
        self.model_type = model_type
        self.verbose = verbose

        if verbose:
            print("Instantiated C++ WaveRNN Vocoder wrapper for model: ", self.model_fpath)

        # For high speed inference we need to make best utilization of CPU resources and threads.
        self._processing_thread_wrappers = []

    def load(self, max_threads=None):
        """
        Determines best processing setup for high speed C++ inference and sets everything up.
        """
        # Get max availiable physical threads of this CPU
        # TODO: Find out if this is faster when using physical vs logical
        cpus = psutil.cpu_count(logical=False)
        if max_threads is not None and max_threads < cpus:
            cpus = max_threads

        # Init wrapper list
        self._processing_thread_wrappers = []
        for tID in range(cpus):
            # Init vocoder wrapper for the model file
            vocoder = WaveRNNVocoder.Vocoder()
            vocoder.loadWeights(str(self.model_fpath))
            # Append it tot the wrapper list
            self._processing_thread_wrappers.append(vocoder)

    def vocode_mel(self, mel, normalize=True, progress_callback=None):
        """
        Infers the waveform of a mel spectrogram output by the synthesizer.

        :param mel: The mel from the synthesizer
        :param progress_callback: Callback for notifying caller about processing progress
        :return: vocoded waveform ready to be saved
        """

        if self.model_type == base.MODEL_TYPE_FATCHORD:
            hp_wavernn = wavernn_fatchord
        elif self.model_type == base.MODEL_TYPE_GENEING:
            hp_wavernn = wavernn_geneing
        elif self.model_type == base.MODEL_TYPE_RUNTIMERACER:
            hp_wavernn = wavernn_runtimeracer
        else:
            raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % self.model_type)

        if normalize:
            # Adjust mel range range to [-1, 1] (normalization)
            mel = mel / sp.max_abs_value

        wave_len = mel.shape[1] * sp.hop_size

        output = None
        wrapper_count = len(self._processing_thread_wrappers)
        if wrapper_count == 0:
            raise RuntimeError("No processing thread wrappers. Did you properly load the Vocoder instance? Aborting...")
        elif wrapper_count == 1:
            output = self.vocode_thread(0, mel)
        else:
            """
            Determine optimal size for folding.
            - For a small mel we might not need all avaliable cores.
            - For a larger mel which would have more batch elements than cores, it is more efficient to increase the
              amount of frames for each core so we can compute the whole wav in a single threading cycle.
            """
            min_target = hp_wavernn.gen_target
            min_overlap = hp_wavernn.gen_overlap

            optimal_target = math.ceil(((wave_len - min_overlap) / wrapper_count) - min_overlap)
            if optimal_target < min_target:
                optimal_target = min_target

            # Do the folding
            mels = self.fold_mel_with_overlap(mel, optimal_target, min_overlap)

            # Render using multithreading
            chunks_count = len(mels)
            threads = []
            with ThreadPoolExecutor(max_workers=wrapper_count) as executor:
                for tID in range(chunks_count):
                    threads.append(executor.submit(self.vocode_thread, tID, mels[tID]))

            # Unfold the results
            output = []
            for t in threads:
                output.append(t.result())
            output = np.stack(output, axis=0)
            output = self.unfold_wav_with_overlap(output, optimal_target, min_overlap)

        if hp_wavernn.mu_law:
            # Do MuLaw decode over the whole generated audio for optimal normalization
            output = decode_mu_law(output, 2 ** hp_wavernn.bits, False)
        if sp.preemphasize:
            output = de_emphasis(output)

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * sp.hop_size)
        output = output[:wave_len]
        output[-20 * sp.hop_size:] *= fade_out

        return output

    def vocode_thread(self, tID, chunk):
        if self.verbose:
            print("Starting libwavernn processing thread ", tID)
        return self._processing_thread_wrappers[tID].melToWav(chunk)

    def fold_mel_with_overlap(self, mel, target, overlap):
        # This folding happens before upsampling on the raw mel
        mel_len = mel.shape[1]
        mel_target = math.ceil(target / sp.hop_size)
        mel_overlap = math.ceil(overlap / sp.hop_size)

        # Calculate variables needed
        num_folds = (mel_len - mel_overlap) // (mel_target + mel_overlap)
        extended_len = num_folds * (mel_overlap + mel_target) + mel_overlap
        remaining = mel_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = mel_target + 2 * mel_overlap - remaining
            total = mel_len + padding
            # Apply padding to the mel
            padded_mel = np.zeros(shape=(mel.shape[0], total), dtype=np.float32)
            padded_mel[:, :mel_len] = mel
            mel = padded_mel

        # Get the values for the folded mel parts
        folded = []
        for i in range(num_folds):
            start = i * (mel_target + mel_overlap)
            end = start + mel_target + 2 * mel_overlap
            folded_elem = mel[:, start:end]
            folded.append(folded_elem)

        return folded

    def unfold_wav_with_overlap(self, wav, target, overlap):

        num_folds, length = wav.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

        # Apply the gain to the overlap samples
        wav[:, :overlap] *= fade_in
        wav[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += wav[i]

        return unfolded

    def setRandomSeed(self, seed):
        if len(self._processing_thread_wrappers) == 0:
            return
        for w in self._processing_thread_wrappers:
            w.setRandomSeed(seed)