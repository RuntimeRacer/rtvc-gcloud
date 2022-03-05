import flask

app = flask.Flask(__name__)

# Hello World simple test function
def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    if request.args and 'message' in request.args:
        return request.args.get('message')
    elif request_json and 'message' in request_json:
        return request_json['message']
    else:
        return f'Hello World!'


# entry point of this gcloud function
# Expects requests to be sent via POST method.
# In case of a non-POST Request it will respond with get_version().
@app.route("/")
@app.route("/encode")
@app.route("/synthesize")
@app.route("/vocode")
@app.route("/render")
def handle_request(request: flask.Request):
    # Parse request data anyways
    method = request.method
    request_data = request.get_json()
    if method != 'POST' or not request_data:
        if request.args:
            return flask.make_response(get_version(method, request.args))
        elif request_data:
            return flask.make_response(get_version(method, request_data))
        else:
            return flask.make_response(get_version(method))

    # Get route and forward request
    if 'encode' in request.url_rule.rule:
        return flask.make_response(process_encode_request(request_data))
    if 'synthesize' in request.url_rule.rule:
        return flask.make_response(process_synthesize_request(request_data))
    if 'vocode' in request.url_rule.rule:
        return flask.make_response(process_vocode_request(request_data))
    if 'render' in request.url_rule.rule:
        return flask.make_response(process_render_request(request_data))
    else:
        return flask.make_response(get_version(method, request_data))

# process_encode_request
# Input params:
# - binary wav file of speaker voice
# Returns:
# - speaker embedding generated from input wav
# - embedding graph generated from input wav
def process_encode_request(request_data):
    # TODO
    return {"success":"encoder function triggered"}

# process_synthesize_request
# Input params:
# - speaker embedding
# - text to synthesize via embedding
# (- fixed neural network seed)
# Returns:
# - binary spectogram of synthesized text
def process_synthesize_request(request_data):
    # TODO
    return {"success":"synthesizer function triggered"}

# process_vocode_request
# Input params:
# - binary spectogram of synthesized text
# (- fixed neural network seed)
# Returns:
# - binary wav file of
def process_vocode_request(request_data):
    return {"success":"vocoder function triggered"}

# process_render_request -> Performs synthesize & vocode in a single step
# Input params:
# - speaker embedding
# - text to synthesize via embedding
# (- fixed neural network seed)
# Returns:
# - binary wav file of
def process_render_request(request_data):
    return {"success":"vocoder function triggered"}

# get_version returns basic info on this gcloud function
def get_version(request_method=None, request_args=None):
    response = {
        "function_name": __name__,
        "version": "0.1-beta",
        "usage": ""
    }
    if request_method != None:
        response["request_method"] = request_method
    if request_args != None:
        response["request_args"] = request_args
    return response