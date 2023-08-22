import numpy as np
from config.hparams import sp, wavernn_fatchord, wavernn_geneing, wavernn_runtimeracer, multiband_melgan
from vocoder.parallel_wavegan.models.melgan import MelGANGenerator, MelGANMultiScaleDiscriminator
from vocoder.parallel_wavegan.layers.pqmf import PQMF
from vocoder.parallel_wavegan.losses.adversarial_loss import GeneratorAdversarialLoss, DiscriminatorAdversarialLoss
from vocoder.parallel_wavegan.losses.feat_match_loss import FeatureMatchLoss
from vocoder.parallel_wavegan.losses.stft_loss import MultiResolutionSTFTLoss
from vocoder.parallel_wavegan import optimizers
from vocoder.wavernn.models.fatchord_version import WaveRNN as WaveRNNFatchord
from vocoder.wavernn.models.geneing_version import WaveRNN as WaveRNNGeneing
from vocoder.wavernn.models.runtimeracer_version import WaveRNN as WaveRNNRuntimeRacer

# Vocoder Types
VOC_TYPE_CPP = 'libwavernn'
VOC_TYPE_PYTORCH = 'pytorch'

# Vocoder Models
MODEL_TYPE_FATCHORD = 'fatchord-wavernn'
MODEL_TYPE_GENEING = 'geneing-wavernn'
MODEL_TYPE_RUNTIMERACER = 'runtimeracer-wavernn'
MODEL_TYPE_MULTIBAND_MELGAN = 'multiband-melgan'


# init_voc_model creates a model object and a pruner object (if applicable) for given hyperparameters
def init_voc_model(
        model_type,
        device,
        override_hp_fatchord=None,
        override_hp_geneing=None,
        override_hp_runtimeracer=None,
        override_hp_melgan=None
):
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

    elif model_type == MODEL_TYPE_MULTIBAND_MELGAN:
        hparams = multiband_melgan
        if override_hp_melgan is not None:
            hparams = override_hp_melgan

        # Determine which Generator to use
        if hparams.generator_type == "MelGANGenerator":
            generator = MelGANGenerator(
                in_channels=hparams.generator_in_channels,
                out_channels=hparams.generator_out_channels,
                kernel_size=hparams.generator_kernel_size,
                channels=hparams.generator_channels,
                upsample_scales=hparams.generator_upsample_scales,
                stack_kernel_size=hparams.generator_stack_kernel_size,
                stacks=hparams.generator_stacks,
                use_weight_norm=hparams.generator_use_weight_norm,
                use_causal_conv=hparams.generator_use_causal_conv,
            ).to(device)
        else:
            raise NotImplementedError("Invalid generator of type '%s' provided. Aborting..." % hparams.generator_type)

        # Determine which Discriminator to use
        if hparams.discriminator_type == "MelGANMultiScaleDiscriminator":
            discriminator = MelGANMultiScaleDiscriminator(
                in_channels=hparams.discriminator_in_channels,
                out_channels=hparams.discriminator_out_channels,
                scales=hparams.discriminator_scales,
                downsample_pooling=hparams.discriminator_downsample_pooling,
                downsample_pooling_params=hparams.discriminator_downsample_pooling_params,
                kernel_sizes=hparams.discriminator_kernel_sizes,
                channels=hparams.discriminator_channels,
                max_downsample_channels=hparams.discriminator_max_downsample_channels,
                downsample_scales=hparams.discriminator_downsample_scales,
                nonlinear_activation=hparams.discriminator_nonlinear_activation,
                nonlinear_activation_params=hparams.discriminator_nonlinear_activation_params,
                use_weight_norm=hparams.discriminator_use_weight_norm,
            ).to(device)
        else:
            raise NotImplementedError("Invalid discriminator of type '%s' provided. Aborting..." % hparams.discriminator_type)

        # GAN Models consist of 2 models actually, so we use a dict mapping here instead.
        model = {
            "model_type": model_type,
            "generator": generator,
            "discriminator": discriminator
        }
    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % model_type)

    return model, pruner


# init_criterion creates a criterion object for given model and hyperparameters
def init_criterion(
        model_type,
        device,
        override_hp_melgan=None
):
    criterion = None

    if model_type == MODEL_TYPE_MULTIBAND_MELGAN:
        hparams = multiband_melgan
        if override_hp_melgan is not None:
            hparams = override_hp_melgan

        # Adversarial losses - These just use their default setup with MB-MelGAN
        criterion = {
            "gen_adv": GeneratorAdversarialLoss().to(device),
            "dis_adv": DiscriminatorAdversarialLoss().to(device)
        }

        # STFT Losses
        if hparams.use_stft_loss:
            criterion["stft"] = MultiResolutionSTFTLoss(
                **hparams.stft_loss_params,
            ).to(device)
        if hparams.use_subband_stft_loss:
            assert hparams.generator_out_channels > 1
            criterion["sub_stft"] = MultiResolutionSTFTLoss(
                **hparams.stft_loss_params,
            ).to(device)
        if hparams.use_feat_match_loss:
            # Not used for MB-MelGAN, and seems to rely on default values
            criterion["feat_match"] = FeatureMatchLoss().to(device)

        # define special module for subband processing
        if hparams.generator_out_channels > 1:
            criterion["pqmf"] = PQMF(
                subbands=hparams.generator_out_channels,
            ).to(device)

    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % model_type)

    return criterion


# init_optimizers creates an optimizer object for given model and hyperparameters
def init_optimizers(
        model,
        model_type,
        device,
        override_hp_melgan=None
):
    optimizer = None

    if model_type == MODEL_TYPE_MULTIBAND_MELGAN:
        hparams = multiband_melgan
        if override_hp_melgan is not None:
            hparams = override_hp_melgan

        generator_optimizer_class = getattr(
            optimizers,
            # keep compatibility
            hparams.generator_optimizer_type
        )
        discriminator_optimizer_class = getattr(
            optimizers,
            # keep compatibility
            hparams.discriminator_optimizer_type
        )

        optimizer = {
            "generator": generator_optimizer_class(
                model["generator"].parameters(),
                **hparams.generator_optimizer_params,
            ),
            "discriminator": discriminator_optimizer_class(
                model["discriminator"].parameters(),
                **hparams.discriminator_optimizer_params,
            ),
        }

    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % model_type)

    return optimizer


def get_model_type(model):
    if isinstance(model, WaveRNNFatchord):
        return MODEL_TYPE_FATCHORD
    elif isinstance(model, WaveRNNGeneing):
        return MODEL_TYPE_GENEING
    elif isinstance(model, WaveRNNRuntimeRacer):
        return MODEL_TYPE_RUNTIMERACER
    elif isinstance(model, dict) and "model_type" in model:
        # For composite models
        return model["model_type"]
    else:
        raise NotImplementedError("Provided object is not a valid vocoder model.")
