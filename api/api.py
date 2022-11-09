import argparse
import os
from flask import Flask, request, abort, jsonify, send_from_directory
from pathlib import Path


app = Flask(__name__)

BASE_DIR = Path(__name__).resolve().parent.parent
DOWNLOAD_DIRECTORY = BASE_DIR / 'tflite'

if not os.path.exists(DOWNLOAD_DIRECTORY):
    BASE_DIR = Path(__name__).resolve().parent
    DOWNLOAD_DIRECTORY = BASE_DIR / 'tflite'


@app.route('/')
def index():
    html = """
            <h3>POC RECONHECIMENTO FACIAL - CRP</h1>
            <hr/>
            <ul>
                <li>list all files to download: <a href="/files">/files</a></li>
                <li>get especifique file: <a href="javascript:void(0)">/get-file/{file:file-name} (GET or POST)</a></li>
            </ul>
            :return:
            """
    return html


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