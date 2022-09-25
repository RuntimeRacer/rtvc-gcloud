import ast
import pprint


class HParams(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def parse(self, string):
        # Overrides hparams from a comma-separated string of name=value pairs
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self


# Global Parameters for multispeaker TTS
sv2tts = HParams(
    speaker_embedding_size=768,  # Dimension for the speaker embedding
)

# Global parameters for signal processing (used in both synthesizer and vocoder models)
sp = HParams(
    sample_rate=16000,
    n_fft=1024,
    num_mels=80,
    hop_size=200,  # Tacotron uses 12.5 ms frame shift (set to sample_rate * 0.0125)
    win_size=800,  # Tacotron uses 50 ms frame length (set to sample_rate * 0.050)
    fmin=40,
    fmax=8000,  # Should not exceed (sample_rate // 2)
    min_level_db=-100,
    ref_level_db=20,
    max_abs_value=4.,  # Gradient explodes if too big, premature convergence if too small.
    preemphasis=0.97,  # Filter coefficient to use if preemphasize is True
    preemphasize=True,
)

# Global Parameters for Data Preprocessing
preprocessing = HParams(
    max_mel_frames=1200,
    rescale=True,
    rescaling_max=0.9,
    synthesis_batch_size=24,
    # For vocoder preprocessing and inference. - Rule of Thumb: 1 unit per GB of VRAM of smallest card

    # Mel Visualization and Griffin-Lim
    signal_normalization=True,
    power=1.5,
    griffin_lim_iters=80,

    # Audio processing options
    allow_clipping_in_normalization=True,  # Used when signal_normalization = True
    clip_mels_length=True,  # If true, discards samples exceeding max_mel_frames
    use_lws=False,  # "Fast spectrogram phase recovery using local weighted sums"
    symmetric_mels=True,  # Sets mel range to [-max_abs_value, max_abs_value] if True, and [0, max_abs_value] if False
    trim_silence=True,  # Use with sample_rate of 16000 for best results
    silence_min_duration_split=0.4,  # Duration in seconds of a silence for an utterance to be split
    utterance_min_duration=0.6,  # Duration in seconds below which utterances are discarded
    trim_start_end_silence=True,  # Whether to trim leading and trailing silence
    trim_silence_top_db=60,  # Threshold in decibels below reference to consider silence for trimming
    pitch_max_freq=600,  # Maximum value for pitch frequency to remove outliers (Common pitch range is about 60-300)

    # Text Preprocessing
    cleaner_names=["english_cleaners"],
    min_text_len=2,
    extract_durations_with_dijkstra=True,

    # Silence tweaks for Prediction models
    silence_prob_shift=0.25,    # Increase probability for silent characters in periods of silence
                                # for better durations during non voiced periods
    silence_threshold=-11,      # normalized mel value below which the voice is considered silent
                                # minimum mel value = -11.512925465 for zeros in the wav array (=log(1e-5),
                                # where 1e-5 is a cutoff value)

    # Attention Scoring during dataset loading
    filter_attention=True,
    min_attention_sharpness=0.5,
    min_attention_alignment=0.95,
)

# Parameters for Tacotron model
tacotron = HParams(
    # Tacotron Text-to-Speech (TTS)
    embed_dims=256,  # Embedding dimension for the graphemes/phoneme inputs
    encoder_dims=128,
    decoder_dims=256,
    postnet_dims=128,
    encoder_K=16,
    lstm_dims=512,
    postnet_K=8,
    num_highways=4,
    dropout=0.5,
    stop_threshold=-3.4,  # Value below which audio generation ends.
    # For example, for a range of [-4, 4], this
    # will terminate the sequence at the first
    # frame that has all values < -3.4

    # Tacotron Training - Progressive epoch training schedule
    #
    # (r, loops, batch_size, init_lr, end_lr)
    #
    # r          = reduction factor - divisor of mel frames synthesized for each decoder iteration
    #              (lesser value => higher resolution and more precise training, but also higher step time)
    # loops      = iteration loops over whole dataset
    # batch_size = amount of dataset items to train on per step
    #
    # Learning rate is applying Stochastic Gradient Descent with Restarts (SGDR)
    # (https://markkhoffmann.medium.com/exploring-stochastic-gradient-descent-with-restarts-sgdr-fa206c38a74e)
    # init_lr    = learning rate at the begin of the epoch
    # end_lr     = learning rate at the end of the epoch

    tts_schedule=[
        (7, 1, 112, 1e-3, 1e-7),
        (6, 2, 100, 9e-4, 1e-7),
        (5, 4, 88, 8e-4, 1e-7),
        (4, 8, 76, 7e-4, 1e-7),
        (3, 16, 64, 5e-4, 1e-7),
        (2, 16, 44, 4e-4, 1e-7),
        (1, 16, 22, 2e-4, 1e-7)
    ],

    tts_clip_grad_norm=1.0,  # clips the gradient norm to prevent explosion - set to None if not needed

    tts_eval_interval=500,
    # Number of steps between model evaluation (sample generation). Set to -1 to generate after completing epoch, or 0 to disable
    tts_eval_num_samples=1,  # Makes this number of samples
)

# Parameters for ForwardTacotron model
forward_tacotron = HParams(
    # Forward-Tacotron Text-to-Speech (TTS)
    embed_dims=256,
    series_embed_dims=64,

    # Duration Predictor
    duration_conv_dims=256,
    duration_rnn_dims=64,
    duration_dropout=0.5,

    # Pitch Predictor
    pitch_conv_dims=256,
    pitch_rnn_dims=128,
    pitch_dropout=0.5,
    pitch_strength=1.,  # set to 0 if you want no pitch conditioning

    # Energy Predictor
    energy_conv_dims=256,
    energy_rnn_dims=64,
    energy_dropout=0.5,
    energy_strength=1.,  # set to 0 if you want no energy conditioning

    # Prenet (aka. Encoder)
    prenet_dims=256,
    prenet_k=16,
    prenet_num_highways=4,
    prenet_dropout=0.5,

    # LSTM
    rnn_dims=512,

    # Postnet (aka. Decoder)
    postnet_dims=256,
    postnet_k=8,
    postnet_num_highways=4,
    postnet_dropout=0.,

    # Forward Tacotron Training
    #
    # (loops, batch_size, init_lr, end_lr)
    #
    # loops      = iteration loops over whole dataset
    # batch_size = amount of dataset items to train on per step
    #
    # Learning rate is applying Stochastic Gradient Descent with Restarts (SGDR)
    # (https://markkhoffmann.medium.com/exploring-stochastic-gradient-descent-with-restarts-sgdr-fa206c38a74e)
    # init_lr    = learning rate at the begin of the epoch
    # end_lr     = learning rate at the end of the epoch
    tts_schedule=[(1, 16, 1e-3, 5e-4),
                  (2, 24, 5e-4, 5e-4),
                  (4, 32, 5e-4, 5e-4),
                  (8, 40, 5e-4, 5e-4),
                  (16, 48, 5e-4, 5e-4),
                  (32, 48, 5e-4, 5e-4),
                  (64, 48, 5e-4, 5e-4)],

    duration_loss_factor=0.1,
    pitch_loss_factor=0.1,
    energy_loss_factor=0.1,
    pitch_zoneout=0.,
    energy_zoneout=0.,
    clip_grad_norm=1.0,  # clips the gradient norm to prevent explosion - set to None if not needed

    # Model Evaluation
    eval_interval=500,
    # Number of steps between model evaluation (sample generation). Set to -1 to generate after completing epoch, or 0 to disable
    eval_num_samples=1,  # Makes this number of samples
)

# Parameters for FastPitch model
fast_pitch = HParams(
    
)

# Parameters for WaveRNN Vocoder
wavernn = HParams(
    # Model
    mode='RAW',  # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
    bits=9,  # bit depth of signal
    mu_law=True,  # Recommended to suppress noise if using raw bits in hp.voc_mode
    upsample_factors=(5, 5, 8),  # NB - this needs to correctly factorise hop_length

    rnn_dims=512,
    fc_dims=512,
    compute_dims=128,
    res_out_dims=128,
    res_blocks=10,

    # WaveRNN Training
    pad=2,  # this will pad the input so that the resnet can 'see' wider than input length
    seq_len=sp.hop_size * 5,  # must be a multiple of hop_length

    # Progressive training schedule
    # (loops, init_lr, final_lr, batch_size)
    # loops = amount of loops through the dataset per epoch
    # init_lr = inital sgdr learning rate
    # final_lr = amount of loops through the dataset per epoch
    # batch_size = Size of the batches used for inference. Rule of Thumb: Max. 12 units per GB of VRAM of smallest card.
    voc_tts_schedule=[
        (1, 5e-3, 1e-3, 120),
        (2, 1e-3, 5e-4, 160),
        (4, 5e-4, 1e-4, 200),
        (256, 1e-4, 1e-4, 240),
        (256, 1e-4, 5e-5, 280),
        (256, 5e-5, 1e-5, 320),
        (256, 1e-5, 1e-5, 360),
    ],

    # Anomaly detection in Training
    anomaly_detection=False,
    # Enables Loss anomaly detection. Dataloader will collect more metadata. Reduces training Performance by ~20%.
    anomaly_trigger_multiplier=6,
    # Threshold for raising anomaly detection. It is a Multiplier of average loss change

    # Generating / Synthesizing
    gen_at_checkpoint=5,  # number of samples to generate at each checkpoint
    gen_batched=True,  # very fast (realtime+) single utterance batched generation
    gen_target=3000,  # target number of samples to be generated in each batch entry
    gen_overlap=1500,  # number of samples for crossfading between batches
)
