# APPLICATION
BACKEND_VERSION = '1'

# MODEL PATHS
ENCODER_PATH = 'models/encoder.pt'
SYNTHESIZER_PATH = 'models/synthesizer.pt'
VOCODER_PATH = 'models/vocoder.pt'
VOCODER_BINARY_PATH = 'models/vocoder.bin'
VOICEFIXER_ANALYZER_PATH = 'models/vf_analyzer.pt'
VOICEFIXER_VOCODER_PATH = 'models/vf_vocoder.pt'

# MISC
INPUT_MIN_LENGTH_SECONDS = 0.5
INPUT_MAX_LENGTH_SECONDS = 15

# ERROR CODES
ERROR_CLIENT_TOKEN_INVALID = 'error_client_token_invalid'
ERROR_NO_AUDIO_PROVIDED = 'error_no_speaker_audio_provided'
ERROR_NO_EMBEDDING_PROVIDED = 'error_no_speaker_embedding_provided'
ERROR_NO_TEXT_PROVIDED = 'error_no_speaker_embedding_provided'
ERROR_NO_SYNTHESIZED_DATA_PROVIDED = 'error_no_synthesized_data_provided'
ERROR_SAMPLE_URL_RESPONSE = 'error_sample_url_response'
ERROR_SAMPLE_DURATION_INVALID = 'error_sample_duration_invalid'
ERROR_SAMPLE_DATA_INVALID = 'error_sample_data_invalid'
ERROR_EMBEDDING_OR_TEXT_INVALID = 'error_embedding_or_text_invalid'
ERROR_SYNTHESIS_DATA_INVALID = 'error_synthesis_data_invalid'
ERROR_SEED_INVALID = 'error_seed_invalid'
ERROR_SPEED_MODIFIER_INVALID = 'error_speed_modifier_invalid'
ERROR_PITCH_MODIFIER_INVALID = 'error_pitch_modifier_invalid'
ERROR_ENERGY_MODIFIER_INVALID = 'error_energy_modifier_invalid'
ERROR_DATA_URL_INVALID = 'error_sample_url_invalid'
ERROR_IMAGE_DATA_INVALID = 'error_image_data_invalid'
ERROR_TEXTS_NOT_A_LIST = 'error_texts_not_a_list'
ERROR_TEXTS_EMPTY = 'error_texts_empty'
ERROR_BATCHING_AMOUNT_EXCEEDED = 'error_batching_amount_exceeded'
ERROR_VOICEFIXER_NOT_FOUND = 'error_voicefixer_not_found'
ERROR_ENCODER_NOT_FOUND = 'error_encoder_not_found'
ERROR_SYNTHESIZER_NOT_FOUND = 'error_synthesizer_not_found'
ERROR_VOCODER_NOT_FOUND = 'error_vocoder_not_found'
ERROR_WHISPER_NOT_FOUND = 'error_whisper_not_found'
ERROR_PROFILING_DISABLED = 'error_profiling_disabled'
