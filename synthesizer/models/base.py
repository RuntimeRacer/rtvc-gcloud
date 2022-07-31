from config.hparams import forward_tacotron as hp_forward_tacotron
from config.hparams import sp, sv2tts
from config.hparams import tacotron as hp_tacotron
from synthesizer.models.forward_tacotron import ForwardTacotron
from synthesizer.models.tacotron import Tacotron
from synthesizer.utils.symbols import symbols

# Synthesizer Models
MODEL_TYPE_TACOTRON = 'tacotron'
MODEL_TYPE_FORWARD_TACOTRON = 'forward-tacotron'


def init_syn_model(model_type, device, override_hp_tacotron=None, override_hp_forward_tacotron=None):
    model = None
    if model_type == MODEL_TYPE_TACOTRON:
        hparams = hp_tacotron
        if override_hp_tacotron is not None:
            hparams = override_hp_tacotron

        model = Tacotron(
            embed_dims=hparams.embed_dims,
            num_chars=len(symbols),
            encoder_dims=hparams.encoder_dims,
            decoder_dims=hparams.decoder_dims,
            n_mels=sp.num_mels,
            fft_bins=sp.num_mels,
            postnet_dims=hparams.postnet_dims,
            encoder_K=hparams.encoder_K,
            lstm_dims=hparams.lstm_dims,
            postnet_K=hparams.postnet_K,
            num_highways=hparams.num_highways,
            dropout=hparams.dropout,
            stop_threshold=hparams.stop_threshold,
            speaker_embedding_size=sv2tts.speaker_embedding_size
        ).to(device)
    elif model_type == MODEL_TYPE_FORWARD_TACOTRON:
        hparams = hp_forward_tacotron
        if override_hp_tacotron is not None:
            hparams = override_hp_forward_tacotron

        model = ForwardTacotron(
            embed_dims=hparams.embed_dims,
            series_embed_dims=hparams.series_embed_dims,
            num_chars=len(symbols),
            n_mels=sp.num_mels,
            durpred_conv_dims=hparams.duration_conv_dims,
            durpred_rnn_dims=hparams.duration_rnn_dims,
            durpred_dropout=hparams.duration_dropout,
            pitch_conv_dims=hparams.pitch_conv_dims,
            pitch_rnn_dims=hparams.pitch_rnn_dims,
            pitch_dropout=hparams.pitch_dropout,
            pitch_strength=hparams.pitch_strength,
            energy_conv_dims=hparams.energy_conv_dims,
            energy_rnn_dims=hparams.energy_rnn_dims,
            energy_dropout=hparams.energy_dropout,
            energy_strength=hparams.energy_strength,
            prenet_dims=hparams.prenet_dims,
            prenet_k=hparams.prenet_k,
            prenet_num_highways=hparams.prenet_num_highways,
            prenet_dropout=hparams.prenet_dropout,
            rnn_dims=hparams.rnn_dims,
            postnet_dims=hparams.postnet_dims,
            postnet_k=hparams.postnet_k,
            postnet_num_highways=hparams.postnet_num_highways,
            postnet_dropout=hparams.postnet_dropout,
            speaker_embed_dims=sv2tts.speaker_embedding_size
        ).to(device)
    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % model_type)

    return model


def get_model_train_elements(model_type):
    train_elements = []
    if model_type == MODEL_TYPE_TACOTRON:
        train_elements = ["mel", "embed"]
    elif model_type == MODEL_TYPE_FORWARD_TACOTRON:
        train_elements = ["mel", "embed", "duration", "attention", "alignment", "phoneme_pitch", "phoneme_energy"]
    else:
        raise NotImplementedError("Invalid model of type '%s' provided. Aborting..." % model_type)
    return train_elements


def get_model_type(model):
    if isinstance(model, Tacotron):
        return MODEL_TYPE_TACOTRON
    elif isinstance(model, ForwardTacotron):
        return MODEL_TYPE_FORWARD_TACOTRON
    else:
        raise NotImplementedError("Provided object is not a valid synthesizer model.")
