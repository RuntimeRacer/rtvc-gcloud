# Real-Time Voice-Cloning (RTVC) GCloud demo script
# (c) @RuntimeRacer 2022 (https://github.com/RuntimeRacer)
#
# This script allows you to generate a speaker embedding from an existing .wav file
# and generate synthesized speech from it.
# It will also save images for the speaker embedding, the mel spectogram of the
#
# Bear in mind that this version of the script is super simple and also the performance of the GCloud backend might
# not be as pleasant as intended.
#
# I will be working on improvements of the performance of the backend, as well as making this set of scripts
# more useable.

import base64
import os
import pathlib
import sys
import funcs


if len(sys.argv) < 4:
    print("Usage: python util_demo.py $INPUT_FILE $TEXT $OUTPUT_FILE")
    exit()

# Parse params
input_file = sys.argv[1]
text = sys.argv[2]
output_file = sys.argv[3]

if len(input_file) == 0 or len(output_file) == 0 or len(text) == 0:
    print("Invalid params")
    exit()

# Read service env var
SERVICE_URL = os.environ["SERVICE_URL"]

# Read in input file
input_base64 = funcs.base64_from_file(input_file)

# Create Request to get embedding
encode_result = funcs.send_encode_request(SERVICE_URL, input_base64)
embed = encode_result["embed"]
embed_graph = encode_result["embed_graph"]
embed_mel_graph = encode_result["embed_mel_graph"]

# Save images returned by encoding process
funcs.save_b64_image(embed_graph, input_file+".embed_graph.png")
funcs.save_b64_image(embed_mel_graph, input_file+".embed_mel_graph.png")

# Create request to synthesize a spectogram
text_b64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
syn_result = funcs.send_synthesize_request(SERVICE_URL, embed, text_b64)
synthesized = syn_result["synthesized"]
mel_graph = syn_result["synthesized_mel_graph"]

# Save image of mel graph
funcs.save_b64_image(mel_graph, output_file+".synthesized_mel_graph.png")

# Create request to vocode synthesized spectogram
voc_result = funcs.send_vocode_request(SERVICE_URL, synthesized)
generated_wav = voc_result["generated_wav"]

# Save generated wav to output file
wav_bytes = funcs.base64_to_file(generated_wav.encode('utf-8'), output_file)







