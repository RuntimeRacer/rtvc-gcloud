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

# Parameters for fatchord's WaveRNN Vocoder
wavernn_fatchord = HParams(
    # Model
    mode='RAW',  # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
    bits=10,  # bit depth of signal
    mu_law=True,  # Recommended to suppress noise if using raw bits in hp.voc_mode
    upsample_factors=(5, 5, 8),  # NB - this needs to correctly factorise hop_length

    rnn_dims=512,
    fc_dims=512,
    compute_dims=128,
    res_out_dims=32*4, #aux output is fed into 2 downstream nets
    res_blocks=10,

    # WaveRNN Training
    pad=2,  # this will pad the input so that the resnet can 'see' wider than input length
    seq_len=sp.hop_size * 5,  # must be a multiple of hop_length
    # seq_len_factor can be adjusted to increase training sequence length (will increase GPU usage)

    # MOL Training params
    num_classes=65536,
    log_scale_min=-32.23619130191664,  # = float(np.log(1e-14))
    # log_scale_min=-16.11809565095831,  # = float(np.log(1e-7))

    # Progressive training schedule
    # (loops, init_lr, final_lr, batch_size)
    # loops = amount of loops through the dataset per epoch
    # init_lr = inital sgdr learning rate
    # final_lr = amount of loops through the dataset per epoch
    # batch_size = Size of the batches used for inference. Rule of Thumb: Max. 12 units per GB of VRAM of smallest card.
    voc_tts_schedule=[
        (1, 1e-3, 5e-4, 40),
        (2, 5e-4, 1e-4, 50),
        (4, 1e-4, 1e-4, 60),
        (8, 1e-4, 1e-4, 70),
        (16, 1e-4, 1e-4, 80),
        (32, 1e-4, 1e-4, 90),
        (64, 1e-4, 1e-4, 100),
        (128, 1e-4, 5e-5, 110),
        (256, 5e-5, 5e-5, 120),
        (256, 5e-5, 5e-5, 120),
        (256, 5e-5, 5e-5, 120),
        (256, 5e-5, 5e-5, 120),
    ],

    # sparsification
    use_sparsification=False,
    start_prune=100000,
    prune_steps=100000,
    sparsity_target=0.90,
    sparsity_target_rnn=0.90,
    sparse_group=4,

    # Anomaly detection in Training
    anomaly_detection=False,  # Enables Loss anomaly detection. TODO: If anamaly is detected, continue training from previous backup
    anomaly_trigger_multiplier=6,  # Threshold for raising anomaly detection. It is a Multiplier of average loss change.
    # Remark: Loss explosion can be caused either by bad training data, or by too high learning rate.
    # Explosion due to high learning rate will happen usually early on.
    # Explosion due to bad data randomly happens even at a high training step.
    # If anomalies occur frequently, try to reduce deviation / bad quality data in your dataset.

    # Generating / Synthesizing
    gen_at_checkpoint=5,  # number of samples to generate at each checkpoint
    gen_batched=True,  # very fast (realtime+) single utterance batched generation
    gen_target=3000,  # target number of samples to be generated in each batch entry
    gen_overlap=1500,  # number of samples for crossfading between batches
)

# Parameters for geneing's optimized WaveRNN Vocoder
wavernn_geneing = HParams(
    # Model
    mode='BITS',  # either 'BITS' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
    bits=10,  # bit depth of signal
    mu_law=False,  # Recommended to suppress noise if using raw bits in hp.voc_mode
    upsample_factors=(4, 5, 10),  # NB - this needs to correctly factorise hop_length

    rnn_dims=256,
    fc_dims=128,
    compute_dims=64,
    res_out_dims=32*2, #aux output is fed into 2 downstream nets
    res_blocks=3,

    # WaveRNN Training
    pad=2,  # this will pad the input so that the resnet can 'see' wider than input length
    seq_len=sp.hop_size * 7,  # must be a multiple of hop_length
    # seq_len_factor can be adjusted to increase training sequence length (will increase GPU usage)

    # MOL Training params
    num_classes=256,
    log_scale_min=-32.23619130191664,  # = float(np.log(1e-14))
    #log_scale_min=-16.11809565095831,  # = float(np.log(1e-7))

    # Progressive training schedule
    # (loops, init_lr, final_lr, batch_size)
    # loops = amount of loops through the dataset per epoch
    # init_lr = inital sgdr learning rate
    # final_lr = amount of loops through the dataset per epoch
    # batch_size = Size of the batches used for inference. Rule of Thumb: Max. 12 units per GB of VRAM of smallest card.
    voc_tts_schedule=[
        (0.25, 1e-3, 5e-4, 40),
        (0.50, 5e-4, 1e-4, 60),
        (1, 1e-4, 5e-5, 80),
        (2, 5e-5, 5e-5, 100),
        (4, 5e-5, 5e-5, 110),
        (8, 5e-5, 5e-5, 120),
        (16, 5e-5, 5e-5, 130),
        (32, 5e-5, 5e-5, 140),
        (64, 5e-5, 5e-5, 150),
        (64, 5e-5, 5e-5, 150),
        (64, 5e-5, 5e-5, 150),
        (64, 5e-5, 5e-5, 150),
    ],

    # sparsification
    use_sparsification=False,
    start_prune=100000,
    prune_steps=100000,
    sparsity_target=0.90,
    sparsity_target_rnn=0.90,
    sparse_group=4,

    # Anomaly / Loss explosion detection in Training
    anomaly_detection=False,  # Enables Loss anomaly detection. TODO: If anamaly is detected, continue training from previous backup
    anomaly_trigger_multiplier=6,  # Threshold for raising anomaly detection. It is a Multiplier of average loss change.
    # Remark: Loss explosion can be caused either by bad training data, or by too high learning rate.
    # Explosion due to high learning rate will happen usually early on.
    # Explosion due to bad data randomly happens even at a high training step.
    # If anomalies occur frequently, try to reduce deviation / bad quality data in your dataset.

    # Generating / Synthesizing
    gen_at_checkpoint=5,  # number of samples to generate at each checkpoint
    gen_batched=True,  # very fast (realtime+) single utterance batched generation
    gen_target=3000,  # target number of samples to be generated in each batch entry
    gen_overlap=1500,  # number of samples for crossfading between batches
)

# Parameters for RuntimeRacer's optimized WaveRNN Vocoder
wavernn_runtimeracer = HParams(
    # Model
    mode='RAW',  # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
    bits=10,  # bit depth of signal
    mu_law=True,  # Recommended to suppress noise if using raw bits in hp.voc_mode
    upsample_factors=(5, 5, 8),  # NB - this needs to correctly factorise hop_length

    rnn_dims=256,
    fc_dims=256,
    compute_dims=128,
    res_out_dims=64*2, #aux output is fed into 2 downstream nets
    res_blocks=10,

    # WaveRNN Training
    pad=2,  # this will pad the input so that the resnet can 'see' wider than input length
    seq_len=sp.hop_size * 5,  # must be a multiple of hop_length
    # seq_len_factor can be adjusted to increase training sequence length (will increase GPU usage)

    # MOL Training params
    num_classes=65536,
    log_scale_min=-32.23619130191664,  # = float(np.log(1e-14))
    #log_scale_min=-16.11809565095831,  # = float(np.log(1e-7))

    # Progressive training schedule
    # (loops, init_lr, final_lr, batch_size)
    # loops = amount of loops through the dataset per epoch
    # init_lr = inital sgdr learning rate
    # final_lr = amount of loops through the dataset per epoch
    # batch_size = Size of the batches used for inference. Rule of Thumb: Max. 12 units per GB of VRAM of smallest card.
    voc_tts_schedule=[
        (1, 1e-3, 5e-4, 40),
        (2, 5e-4, 1e-4, 50),
        (4, 1e-4, 1e-4, 60),
        (8, 1e-4, 1e-4, 70),
        (16, 1e-4, 1e-4, 80),
        (32, 1e-4, 1e-4, 90),
        (64, 1e-4, 1e-4, 100),
        (128, 1e-4, 5e-5, 110),
        (256, 5e-5, 5e-5, 120),
        (256, 5e-5, 5e-5, 120),
        (256, 5e-5, 5e-5, 120),
        (256, 5e-5, 5e-5, 120),
    ],

    # sparsification
    use_sparsification=False,
    start_prune=100000,
    prune_steps=100000,
    sparsity_target=0.90,
    sparsity_target_rnn=0.90,
    sparse_group=4,

    # Anomaly / Loss explosion detection in Training
    anomaly_detection=False,  # Enables Loss anomaly detection. TODO: If anamaly is detected, continue training from previous backup
    anomaly_trigger_multiplier=6,  # Threshold for raising anomaly detection. It is a Multiplier of average loss change.
    # Remark: Loss explosion can be caused either by bad training data, or by too high learning rate.
    # Explosion due to high learning rate will happen usually early on.
    # Explosion due to bad data randomly happens even at a high training step.
    # If anomalies occur frequently, try to reduce deviation / bad quality data in your dataset.

    # Generating / Synthesizing
    gen_at_checkpoint=5,  # number of samples to generate at each checkpoint
    gen_batched=True,  # very fast (realtime+) single utterance batched generation
    gen_target=6000,  # target number of samples to be generated in each batch entry
    gen_overlap=1000,  # number of samples for crossfading between batches
)

# Parameters for Multiband MelGAN
multiband_melgan = HParams(
    # Generator
    generator_type="MelGANGenerator",
    generator_in_channels=80,  # Number of input channels.
    generator_out_channels=4,  # Number of output channels.
    generator_kernel_size=7,  # Kernel size of initial and final conv layers.
    generator_channels=384,  # Initial number of channels for conv layers.
    generator_upsample_scales=[5, 5, 2],  # List of Upsampling scales.
    generator_stack_kernel_size=3,  # Kernel size of dilated conv layers in residual stack.
    generator_stacks=4,  # Number of stacks in a single residual stack module.
    generator_use_weight_norm=True,  # Whether to use weight normalization.
    generator_use_causal_conv=False,  # Whether to use causal convolution.

    # Discriminator
    discriminator_type="MelGANMultiScaleDiscriminator",
    discriminator_in_channels=1,  # Number of input channels.
    discriminator_out_channels=1,  # Number of output channels.
    discriminator_scales=3,  # Number of multi-scales.
    discriminator_downsample_pooling="AvgPool1d",  # Pooling type for the input downsampling.
    discriminator_downsample_pooling_params={  # Parameters of the above pooling function.
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "count_include_pad": False,
    },
    discriminator_kernel_sizes=[5, 3],  # List of kernel size.
    discriminator_channels=16,  # Number of channels of the initial conv layer.
    discriminator_max_downsample_channels=512,  # Maximum number of channels of downsampling layers.
    discriminator_downsample_scales=[4, 4, 4],  # List of downsampling scales.
    discriminator_nonlinear_activation="LeakyReLU",  # Nonlinear activation function.
    discriminator_nonlinear_activation_params={"negative_slope": 0.2},  # Parameters of nonlinear activation function.
    discriminator_use_weight_norm=True,  # Whether to use weight norm.

    # STFT Loss settings
    # DEFAULT VALUES for 16-khz vocode
    use_stft_loss=True,
    stft_loss_params={
        "fft_sizes": [512, 1024, 256],  # List of FFT size for STFT-based loss.
        "hop_sizes": [60, 120, 25],  # List of hop size for STFT-based loss
        "win_lengths": [300, 600, 120],  # List of window length for STFT-based loss.
        "window": "hann_window",  # Window function for STFT-based loss
    },
    use_subband_stft_loss=True,
    subband_stft_loss_params={
        "fft_sizes": [192, 342, 86],  # List of FFT size for STFT-based loss.
        "hop_sizes": [15, 30, 5],  # List of hop size for STFT-based loss
        "win_lengths": [75, 150, 30],  # List of window length for STFT-based loss.
        "window": "hann_window",  # Window function for STFT-based loss
    },

    # DEFAULT VALUES for 24-khz vocode
    # use_stft_loss=True,
    # stft_loss_params={
    #     "fft_sizes": [1024, 2048, 512],  # List of FFT size for STFT-based loss.
    #     "hop_sizes": [120, 240, 50],  # List of hop size for STFT-based loss
    #     "win_lengths": [600, 1200, 240],  # List of window length for STFT-based loss.
    #     "window": "hann_window",  # Window function for STFT-based loss
    # },
    # use_subband_stft_loss=True,
    # subband_stft_loss_params={
    #     "fft_sizes": [384, 683, 171],  # List of FFT size for STFT-based loss.
    #     "hop_sizes": [30, 60, 10],  # List of hop size for STFT-based loss
    #     "win_lengths": [150, 300, 60],  # List of window length for STFT-based loss.
    #     "window": "hann_window",  # Window function for STFT-based loss
    # },

    # Adversarial Loss Setting
    use_feat_match_loss=False,  # Whether to use feature matching loss.
    lambda_adv=2.5,  # Loss balancing coefficient for adversarial loss.

    # Training Settings
    batch_size=64, # Size of the batches used for training.
    aux_context_window=0,  # this will pad the input so that the resnet can 'see' wider than input length
    seq_len=sp.hop_size * 40,  # Batch max steps (8000) - must be a multiple of hop_length
    # seq_len can be adjusted to increase training sequence length (will increase GPU usage)
    remove_short_samples=True,  # Whether to remove samples the length of which are less than batch_max_steps.

    ##################################################
    # Progressive training schedules
    ##################################################
    # (loops, init_lr, final_lr, batch_size)
    # loops = amount of loops through the dataset per epoch
    # init_lr = inital sgdr learning rate
    # final_lr = amount of loops through the dataset per epoch
    ##################################################

    # Generator Training Settings
    generator_train_start_after_steps=0,  # Generator Training will start once generator reached this step
    generator_optimizer_type="Adam",
    generator_optimizer_params={
        "eps": 1e-7,
        "weight_decay": 0.0,
        "amsgrad": True
    },
    generator_grad_norm=-1,
    generator_scheduler_type="MultiStepLR",
    generator_tts_schedule=[
        (1, 1e-3, 1e-3),
        (2, 1e-3, 5e-4),
        (4, 5e-4, 5e-4),
        (8, 5e-4, 1e-4),
        (16, 1e-4, 1e-4),
        (32, 1e-4, 5e-5),
        (64, 5e-5, 5e-5),
        (128, 5e-5, 5e-5),
        (256, 5e-5, 5e-5),
        (256, 5e-5, 5e-5),
        (256, 5e-5, 5e-5),
        (256, 5e-5, 5e-5),
    ],

    # Discriminator Training Settings
    discriminator_train_start_after_steps=200000,  # Discriminator Training will start once generator reached this step
    discriminator_optimizer_type="Adam",
    discriminator_optimizer_params={
        "eps": 1e-7,
        "weight_decay": 0.0,
        "amsgrad": True
    },
    discriminator_grad_norm=-1,
    discriminator_scheduler_type="MultiStepLR",
    discriminator_tts_schedule=[
        (1, 1e-3, 1e-3),
        (2, 1e-3, 5e-4),
        (4, 5e-4, 5e-4),
        (8, 5e-4, 1e-4),
        (16, 1e-4, 1e-4),
        (32, 1e-4, 5e-5),
        (64, 5e-5, 5e-5),
        (128, 5e-5, 5e-5),
        (256, 5e-5, 5e-5),
        (256, 5e-5, 5e-5),
        (256, 5e-5, 5e-5),
        (256, 5e-5, 5e-5),
    ],

    # Generating / Synthesizing
    gen_at_checkpoint=5,  # number of samples to generate at each checkpoint
)