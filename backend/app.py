from flask import Flask
from flask import request
import os
from flask import Flask, abort, flash, send_file, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from tests.test_visualize_pitch import runVisualizer
from main import run_main
from flask_cors import CORS



app = Flask(__name__)
CORS(app)
ALLOWED_EXTENSIONS = {'mp3'}
UPLOAD_FOLDER = 'recieved'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/delete', methods=['POST'])
def handle_file_del():
    if request.method == "POST":
        for root, dirs, files in os.walk("output"):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
                    abort(500)
        for root, dirs, files in os.walk("recieved"):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
                    abort(500)
    return jsonify({'message': 'Successfully delete user upload from backend'}), 200
        


@app.route('/upload', methods=['GET', 'POST'])
def handle_file_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            t = runVisualizer(filename)
            run_main(t)
            # return send_file(
            #     "output/sheet_music-1.png",
            #     mimetype="image/png",
            #     as_attachment=True,
            #     download_name="output/sheet_music-1.png"  # requires Flask ≥ 2.0
            # )            
            return send_file("output/sheet_music-1.png", mimetype="image/png")

            # return redirect(url_for('download_file', name=filename))        
    # else:
    #     abort(405)


@app.route('/midi', methods=["GET"])
def handle_midi():
    if request.method == "GET":
        return send_file(
            "output/sheet_music.mid",
            mimetype="audio/midi",
            as_attachment=True,
            download_name="sheet_music.mid"
        )
