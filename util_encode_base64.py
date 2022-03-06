import os
import sys
import base64

# Parse params
input = sys.argv[1]
output = sys.argv[2]

if os.path.exists(input) and output is not None and output != "":
    with open(input, "rb") as file:
        encoded_content = base64.b64encode(file.read())

    with open(output, "wb") as file:
        file.write(encoded_content)
else:
    print("invalid args")
