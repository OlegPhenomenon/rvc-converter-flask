from flask import Flask, request, send_file, jsonify
from pathlib import Path
from scipy.io import wavfile
from rvc.modules.vc.modules import VC
import os
import tempfile

app = Flask(__name__)

vc = VC()
vc.get_vc("/root/rvc_converter_flask/rvc_models/kowalski.pth")

@app.route("/convert", methods=["POST"])
def convert_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
        input_path = temp_in.name
        file.save(input_path)

    try:
        tgt_sr, audio_opt, _, _ = vc.vc_single(1, Path(input_path))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            output_path = temp_out.name
            wavfile.write(output_path, tgt_sr, audio_opt)

        return send_file(output_path, as_attachment=True, download_name="converted.wav", mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
