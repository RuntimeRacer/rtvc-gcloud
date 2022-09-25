import numpy as np
from config.hparams import sp, wavernn_fatchord, wavernn_geneing, wavernn_runtimeracer
from vocoder.models.fatchord_version import WaveRNN as WaveRNNFatchord
from vocoder.models.geneing_version import WaveRNN as WaveRNNGeneing
from vocoder.models.runtimeracer_version import WaveRNN as WaveRNNRuntimeRacer
# from vocoder.pruner import Pruner

# Vocoder Types
VOC_TYPE_CPP = 'libwavernn'
VOC_TYPE_PYTORCH = 'pytorch'

# Vocoder Models
MODEL_TYPE_FATCHORD = 'fatchord-wavernn'
MODEL_TYPE_GENEING = 'geneing-wavernn'
MODEL_TYPE_RUNTIMERACER = 'runtimeracer-wavernn'


def init_voc_model(model_type, device, override_hp_fatchord=None, override_hp_geneing=None, override_hp_runtimeracer=None):
    model = None
    pruner = None
    if model_type == MODEL_TYPE_FATCHORD:
        hparams = wavernn_fatchord
        if override_hp_fatchord is not None:
            hparams = override_hp_fatchord

        # Check to make sure the hop length is correctly factorised
        assert np.cumprod(hparams.upsample_factors)[-1] == sp.hop_size

        model = WaveRNNFatchord(
            rnn_dims=hparams.rnn_dims,
            fc_dims=hparams.fc_dims,
            bits=hparams.bits,
            pad=hparams.pad,
            upsample_factors=hparams.upsample_factors,
            feat_dims=sp.num_mels,
            compute_dims=hparams.compute_dims,
            res_out_dims=hparams.res_out_dims,
            res_blocks=hparams.res_blocks,
            hop_length=sp.hop_size,
            sample_rate=sp.sample_rate,
            mode=hparams.mode,
            pruning=True
        ).to(device)

        # Setup pruner if enabled
        # if hparams.use_sparsification:
        #     pruner = Pruner(hparams.start_prune, hparams.prune_steps, hparams.sparsity_target, hparams.sparse_group)
        #     pruner.update_layers(model.prune_layers, True)
    elif model_type == MODEL_TYPE_GENEING:
        hparams = wavernn_geneing
        if override_hp_geneing is not None:
            hparams = override_hp_geneing

        # Check to make sure the hop length is correctly factorised
        assert np.cumprod(hparams.upsample_factors)[-1] == sp.hop_size

        model = WaveRNNGeneing(
            rnn_dims=hparams.rnn_dims,
            fc_dims=hparams.fc_dims,
            bits=hparams.bits,
            pad=hparams.pad,
            upsample_factors=hparams.upsample_factors,
            feat_dims=sp.num_mels,
            compute_dims=hparams.compute_dims,
            res_out_dims=hparams.res_out_dims,
            res_blocks=hparams.res_blocks,
            hop_length=sp.hop_size,
            sample_rate=sp.sample_rate,
            mode=hparams.mode,
            pruning=True
        ).to(device)

        # Setup pruner if enabled
        # if hparams.use_sparsification:
        #     pruner = Pruner(hparams.start_prune, hparams.prune_steps, hparams.sparsity_target, hparams.sparse_group)
        #     pruner.update_layers(model.prune_layers, True)
    elif model_type == MODEL_TYPE_RUNTIMERACER:
        hparams = wavernn_runtimeracer
        if override_hp_runtimeracer is not None:
            hparams = override_hp_runtimeracer

        # Check to make sure the hop length is correctly factorised
        assert np.cumprod(hparams.upsample_factors)[-1] == sp.hop_size

        model = WaveRNNRuntimeRacer(
            rnn_dims=hparams.rnn_dims,
            fc_dims=hparams.fc_dims,
            bits=hparams.bits,
            pad=hparams.pad,
            upsample_factors=hparams.upsample_factors,
            feat_dims=sp.num_mels,
            compute_dims=hparams.compute_dims,
            res_out_dims=hparams.res_out_dims,
            res_blocks=hparams.res_blocks,
            hop_length=sp.hop_size,
            sample_rate=sp.sample_rate,
            mode=hparams.mode,
            pruning=True
        ).to(device)

        # Setup pruner if enabled
        # if hparams.use_sparsification:
        #     pruner = Pruner(hparams.start_prune, hparams.prune_steps, hparams.sparsity_target, hparams.sparse_group)
        #     pruner.update_layers(model.prune_layers, True)
    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % model_type)

    return model, pruner


def get_model_type(model):
    if isinstance(model, WaveRNNFatchord):
        return MODEL_TYPE_FATCHORD
    elif isinstance(model, WaveRNNGeneing):
        return MODEL_TYPE_GENEING
    elif isinstance(model, WaveRNNRuntimeRacer):
        return MODEL_TYPE_RUNTIMERACER
    else:
        raise NotImplementedError("Provided object is not a valid vocoder model.")
