from flask import Flask
from flask import request


app = Flask(__name__)

@app.route('/upload', methods=['GET', 'POST'])
def handle_file_upload():
    if request.method == 'POST':
        return "<p>This route will be used for uploading!</p>"
    else:
        abort(405)
