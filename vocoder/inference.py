from config.hparams import sp, wavernn_fatchord, wavernn_geneing, wavernn_runtimeracer, multiband_melgan
from vocoder import base
import vocoder.wavernn.libwavernn.inference as libwavernn
import torch

from vocoder.parallel_wavegan.layers import PQMF

_model = None
_model_type = None


def load_model(weights_fpath, voc_type=base.VOC_TYPE_PYTORCH, verbose=True):
    global _model, _device, _model_type

    if voc_type == base.VOC_TYPE_PYTORCH:
        if torch.cuda.is_available():
            _device = torch.device('cuda')
        else:
            _device = torch.device('cpu')

        # Load model weights from provided model path
        checkpoint = torch.load(weights_fpath, map_location=_device)
        _model_type = base.MODEL_TYPE_FATCHORD
        if "model_type" in checkpoint:
            _model_type = checkpoint["model_type"]

        # Init the model
        try:
            _model, _ = base.init_voc_model(_model_type, _device)
            if _model_type == base.MODEL_TYPE_MULTIBAND_MELGAN:
                _model["generator"].eval()
                _model["discriminator"].eval()
                _model["generator"].load_state_dict(checkpoint["model"]["generator"])
                _model["discriminator"].load_state_dict(checkpoint["model"]["discriminator"])

                if verbose:
                    print("Loaded vocoder of model '%s' at path '%s'." % (_model_type, weights_fpath))
                    print("Model has been trained to step %d." % (checkpoint["steps"] if "steps" in checkpoint else 0))
            else:
                _model = _model.eval()
                _model.load_state_dict(checkpoint["model_state"])
                if verbose:
                    print("Loaded vocoder of model '%s' at path '%s'." % (_model_type, weights_fpath))
                    print("Model has been trained to step %d." % (_model.state_dict()["step"]))
        except NotImplementedError as e:
            print(str(e))
            return

    elif voc_type == base.VOC_TYPE_CPP:
        # FIXME: Vocoder type is hacky
        _model = libwavernn.Vocoder(weights_fpath, 'runtimeracer-wavernn', verbose)
        _model.load()
        _model_type = voc_type
        # FIXME: This works because there is only one CPP vocoder implementaion available, but _model_type has
        # FIXME: different meaning depending on context.

        if verbose:
            print("Loaded vocoder of model '%s' at path '%s'." % (_model_type, weights_fpath))

    else:
        raise NotImplementedError("Invalid vocoder of type '%s' provided. Aborting..." % voc_type)


def is_loaded():
    return _model is not None


def infer_waveform(mel, normalize=True, batched=True, target=None, overlap=None):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match
    that of the synthesizer!)

    :param normalize:
    :param batched:
    :param target:
    :param overlap:
    :return:
    """
    if _model is None or _model_type is None:
        raise Exception("Please load Wave-RNN in memory before using it")

    if _model_type == base.VOC_TYPE_CPP:
        wav = _model.vocode_mel(mel=mel, normalize=normalize)
        return wav
    else:
        if _model_type == base.MODEL_TYPE_FATCHORD:
            hparams = wavernn_fatchord
        elif _model_type == base.MODEL_TYPE_GENEING:
            hparams = wavernn_geneing
        elif _model_type == base.MODEL_TYPE_RUNTIMERACER:
            hparams = wavernn_runtimeracer
        elif _model_type == base.MODEL_TYPE_MULTIBAND_MELGAN:
            hparams = multiband_melgan
        else:
            raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % _model_type)

        if _model_type == base.MODEL_TYPE_MULTIBAND_MELGAN:
            # Ensure PQMF
            if _model["generator"].pqmf is None and hparams.generator_out_channels > 1:
                _model["generator"].pqmf = PQMF(
                    subbands=hparams.generator_out_channels,
                )
            # Prepare mel for decoding
            if normalize:
                mel = mel / sp.max_abs_value
            with torch.no_grad():
                wav = _model["generator"].inference(mel.T)
            return wav.squeeze(1)
        else:
            if target is None:
                target = hparams.gen_target
            if overlap is None:
                overlap = hparams.gen_overlap

            if normalize:
                mel = mel / sp.max_abs_value
            mel = torch.from_numpy(mel[None, ...])
            wav = _model.generate(mel, batched, target, overlap, hparams.mu_law, sp.preemphasize)
            return wav


def set_seed(seed):
    if is_loaded() and _model_type == base.VOC_TYPE_CPP:
        _model.setRandomSeed(seed)
    else:
        torch.manual_seed(seed)
