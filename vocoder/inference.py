import torch

from config.hparams import sp
from config.hparams import wavernn as hp_wavernn
from vocoder.models.fatchord_version import WaveRNN

_model = None  # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    global _model, _device
    
    if verbose:
        print("Building Wave-RNN")
    _model = WaveRNN(
        rnn_dims=hp_wavernn.rnn_dims,
        fc_dims=hp_wavernn.fc_dims,
        bits=hp_wavernn.bits,
        pad=hp_wavernn.pad,
        upsample_factors=hp_wavernn.upsample_factors,
        feat_dims=sp.num_mels,
        compute_dims=hp_wavernn.compute_dims,
        res_out_dims=hp_wavernn.res_out_dims,
        res_blocks=hp_wavernn.res_blocks,
        hop_length=sp.hop_size,
        sample_rate=sp.sample_rate,
        mode=hp_wavernn.mode
    )

    if torch.cuda.is_available():
        _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()


def is_loaded():
    return _model is not None


def infer_waveform(mel, normalize=True,  batched=True, target=hp_wavernn.gen_target, overlap=hp_wavernn.gen_overlap):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    if normalize:
        mel = mel / sp.max_abs_value
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(mel, batched, target, overlap, hp_wavernn.mu_law, sp.preemphasize)
    return wav
