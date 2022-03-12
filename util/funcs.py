import json
import os
import base64

import requests


def base64_from_file(path):
    if os.path.exists(path):
        with open(path, "rb") as file:
            encoded_content = base64.b64encode(file.read())
        return encoded_content

def base64_to_file(data, path):
    with open(path, "wb") as file:
        decoded_content = base64.b64decode(data)
        file.write(decoded_content)

def send_encode_request(service_base_url, base64_wav):
    encode_url = service_base_url + '/encode'

    data = {
        "speaker_wav": base64_wav.decode('utf-8')
    }

    print("Sending encode request...")
    response = requests.post(
        url=encode_url,
        json=data
    )

    if response.status_code != 200:
        print("Error returned, status code: %d" % response.status_code)
    print("encode request performed successfully!")

    result = response.json()
    return result

def send_synthesize_request(service_base_url, embedding, text):
    synthesize_url = service_base_url + '/synthesize'

    data = {
        "speaker_embed": embedding,
        "text": text
    }

    print("Sending synthesize request...")
    response = requests.post(
        url=synthesize_url,
        json=data
    )

    if response.status_code != 200:
        print("Error returned, status code: %d" % response.status_code)
    print("synthesize request performed successfully!")

    result = response.json()
    return result

def send_vocode_request(service_base_url, synthesized):
    vocode_url = service_base_url + '/vocode'

    data = {
        "synthesized": synthesized
    }

    print("Sending vocode request...")
    response = requests.post(
        url=vocode_url,
        json=data
    )

    if response.status_code != 200:
        print("Error returned, status code: %d" % response.status_code)
    print("vocode request performed successfully!")

    result = response.json()
    return result

def save_b64_image(data, path):
    data = data.replace('data:image/png;base64,', '')
    with open(path, 'wb') as img:
        img_bytes = base64.b64decode(data.encode('utf-8'))
        img.write(img_bytes)