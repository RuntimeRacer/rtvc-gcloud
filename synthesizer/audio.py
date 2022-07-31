import librosa
import librosa.filters
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile

from config.hparams import preprocessing, sp


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    sf.write(path, wav.astype(np.float32), sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


#From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break
    
    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold
    
    return start, end


def get_hop_size():
    hop_size = sp.hop_size
    if hop_size is None:
        assert sp.frame_shift_ms is not None
        hop_size = int(sp.frame_shift_ms / 1000 * sp.sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, sp.preemphasis, sp.preemphasize))
    S = _amp_to_db(np.abs(D)) - sp.ref_level_db
    
    if preprocessing.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, sp.preemphasis, sp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - sp.ref_level_db
    
    if preprocessing.signal_normalization:
        return _normalize(S)
    return S


def inv_linear_spectrogram(linear_spectrogram):
    """Converts linear spectrogram to waveform using librosa"""
    if preprocessing.signal_normalization:
        D = _denormalize(linear_spectrogram)
    else:
        D = linear_spectrogram
    
    S = _db_to_amp(D + sp.ref_level_db) #Convert back to linear
    
    if preprocessing.use_lws:
        processor = _lws_processor()
        D = processor.run_lws(S.astype(np.float64).T ** preprocessing.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, sp.preemphasis, sp.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** preprocessing.power), sp.preemphasis, sp.preemphasize)


def inv_mel_spectrogram(mel_spectrogram):
    """Converts mel spectrogram to waveform using librosa"""
    if preprocessing.signal_normalization:
        D = _denormalize(mel_spectrogram)
    else:
        D = mel_spectrogram
    
    S = _mel_to_linear(_db_to_amp(D + sp.ref_level_db))  # Convert back to linear
    
    if preprocessing.use_lws:
        processor = _lws_processor()
        D = processor.run_lws(S.astype(np.float64).T ** preprocessing.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, sp.preemphasis, sp.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** preprocessing.power), sp.preemphasis, sp.preemphasize)


def _lws_processor():
    import lws
    return lws.lws(sp.n_fft, get_hop_size(), fftsize=sp.win_size, mode="speech")


def _griffin_lim(S):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(preprocessing.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _stft(y):
    if preprocessing.use_lws:
        return _lws_processor().stft(y).T
    else:
        return librosa.stft(y=y, n_fft=sp.n_fft, hop_length=get_hop_size(), win_length=sp.win_size)


def _istft(y):
    return librosa.istft(y, hop_length=get_hop_size(), win_length=sp.win_size)


##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis():
    assert sp.fmax <= sp.sample_rate // 2
    return librosa.filters.mel(sr=sp.sample_rate, n_fft=sp.n_fft, n_mels=sp.num_mels,
                               fmin=sp.fmin, fmax=sp.fmax)


def _amp_to_db(x):
    min_level = np.exp(sp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S):
    if preprocessing.allow_clipping_in_normalization:
        if preprocessing.symmetric_mels:
            return np.clip((2 * sp.max_abs_value) * ((S - sp.min_level_db) / (-sp.min_level_db)) - sp.max_abs_value,
                           -sp.max_abs_value, sp.max_abs_value)
        else:
            return np.clip(sp.max_abs_value * ((S - sp.min_level_db) / (-sp.min_level_db)), 0, sp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - sp.min_level_db >= 0
    if preprocessing.symmetric_mels:
        return (2 * sp.max_abs_value) * ((S - sp.min_level_db) / (-sp.min_level_db)) - sp.max_abs_value
    else:
        return sp.max_abs_value * ((S - sp.min_level_db) / (-sp.min_level_db))


def _denormalize(D):
    if preprocessing.allow_clipping_in_normalization:
        if preprocessing.symmetric_mels:
            return (((np.clip(D, -sp.max_abs_value,
                              sp.max_abs_value) + sp.max_abs_value) * -sp.min_level_db / (2 * sp.max_abs_value))
                    + sp.min_level_db)
        else:
            return ((np.clip(D, 0, sp.max_abs_value) * -sp.min_level_db / sp.max_abs_value) + sp.min_level_db)
    
    if preprocessing.symmetric_mels:
        return (((D + sp.max_abs_value) * -sp.min_level_db / (2 * sp.max_abs_value)) + sp.min_level_db)
    else:
        return ((D * -sp.min_level_db / sp.max_abs_value) + sp.min_level_db)
