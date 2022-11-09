import argparse
import os
from flask import Flask, request, abort, jsonify, send_from_directory, send_file

app = Flask(__name__)

DOWNLOAD_DIRECTORY = f'{os.getcwd()}/tflite/'

if not os.path.exists(DOWNLOAD_DIRECTORY):
    os.makedirs(DOWNLOAD_DIRECTORY)


@app.route('/')
def index():
    return '<h1>POC RECONHECIMENTO FACIAL CRP</h1>'


@app.route('/files')
def list_files():
    files = []
    for filename in os.listdir(DOWNLOAD_DIRECTORY):
        path = os.path.join(DOWNLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return jsonify(files)


@app.route('/get-file/',  methods=['GET', 'POST'])
def get_file():
    """Download a file."""
    try:
        file = request.args.get('file')
        return send_from_directory(DOWNLOAD_DIRECTORY, file, as_attachment=True)
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=5000,
        help='Port of serving api')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=True)