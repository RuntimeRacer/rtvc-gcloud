"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""

_pad        = "_"
_punctuation = '!\'\"(),-.:;? '
_eos        = "~"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + list(_punctuation) #+ _arpabet
# Required for duration prediction
silent_phonemes_indices = [i for i, p in enumerate(symbols) if p in _pad + _punctuation]