import flask


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
# Expects Utterance requests to be sent via POST method.
# In case of a non-POST Request it will respond with get_version().
def handle_request(request: flask.Request):
    # Parse request data anyways
    request_data = request.get_json()
    if request.method != 'POST':
        if request.args:
            return flask.make_response(get_version(request.args))
        elif request_data:
            return flask.make_response(get_version(request_data))
        else:
            return flask.make_response(get_version())

    # Parse POST data
    request_data = request_data
    return flask.make_response(request_data)

# get_version returns basic info on this gcloud function
def get_version(request_args=None):
    response = {
        "function_name": "rtvc-gcloud-encoder",
        "version": "0.1-beta",
        "usage": ""
    }
    if request_args != None:
        response["request_args"] = request_args
    return response