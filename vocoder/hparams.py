from synthesizer.hparams import hparams as _syn_hp


# Audio settings------------------------------------------------------------------------
# Match the values of the synthesizer
sample_rate = _syn_hp.sample_rate
n_fft = _syn_hp.n_fft
num_mels = _syn_hp.num_mels
hop_length = _syn_hp.hop_size
win_length = _syn_hp.win_size
fmin = _syn_hp.fmin
min_level_db = _syn_hp.min_level_db
ref_level_db = _syn_hp.ref_level_db
mel_max_abs_value = _syn_hp.max_abs_value
preemphasis = _syn_hp.preemphasis
apply_preemphasis = _syn_hp.preemphasize

bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode
                                    # below


# WAVERNN / VOCODER --------------------------------------------------------------------------------
voc_mode = 'MOL'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_upsample_factors = (5, 5, 8)    # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 360                # Rule of Thumb: 12 units per GB of VRAM of smallest card
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider 
                                    # than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length

# Anomaly detection
voc_anomaly_detection = False      # Enables Loss anomaly detection. Dataloader will collect more metadata.
                                   # Reduces training Performance by ~20%
voc_anomaly_trigger_multiplier = 6 # Threshold for raising anomaly detection. It is a Multiplier of average loss change

# Progressive training schedule
# (loops, init_lr, final_lr)
# loops = amount of loops through the dataset per epoch
# init_lr = inital sgdr learning rate
# final_lr = amount of loops through the dataset per epoch
voc_tts_schedule=[
    (2, 5e-3, 1e-3),
    (8, 1e-3, 5e-4),
    (32, 5e-4, 1e-4),
    (1024, 1e-4, 1e-4),
    (128, 1e-4, 5e-5),
    (512, 5e-5, 5e-5),
    (64, 5e-5, 1e-5),
    (256, 1e-5, 1e-5),
]

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 16000                  # target number of samples to be generated in each batch entry
voc_overlap = 800                   # number of samples for crossfading between batches
