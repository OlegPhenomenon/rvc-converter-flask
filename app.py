from flask import Flask, request, send_file, jsonify
from pathlib import Path
import os
import tempfile
from dotenv import load_dotenv
from rvc_python.infer import RVCInference

# curl -X POST \
#   -F "file=@input.wav" \
#   -F "speaker_index=SPEAKER_01" \
#   -F "gender=female" \
#   http://localhost:5000/convert \
#   --output converted.wav

load_dotenv()

app = Flask(__name__)

# Initialize RVC with CUDA if available
device = "cuda:0" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"
rvc = RVCInference(device=device)

# Define available models for different genders
AVAILABLE_MODELS = {
    "male": {
        "SPEAKER_01": "models/male_voice_1.pth",
        "SPEAKER_02": "models/male_voice_2.pth",
        "SPEAKER_03": "models/male_voice_3.pth"
    },
    "female": {
        "SPEAKER_01": "models/female_voice_1.pth",
        "SPEAKER_02": "models/female_voice_2.pth",
        "SPEAKER_03": "models/female_voice_3.pth"
    }
}

# Load default model on startup
DEFAULT_MODEL = AVAILABLE_MODELS["male"]["SPEAKER_01"]
rvc.load_model(DEFAULT_MODEL)

@app.route("/convert", methods=["POST"])
def convert_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Get speaker index and gender from request
    speaker_index = request.form.get("speaker_index", "SPEAKER_01")  # Default SPEAKER_01
    gender = request.form.get("gender", "male").lower()  # Default male

    # Select appropriate model based on gender and speaker index
    if gender not in AVAILABLE_MODELS or speaker_index not in AVAILABLE_MODELS[gender]:
        return jsonify({"error": "Invalid gender or speaker index"}), 400

    model_path = AVAILABLE_MODELS[gender][speaker_index]
    rvc.load_model(model_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
        input_path = temp_in.name
        file.save(input_path)

    try:
        output_path = input_path + "_converted.wav"
        
        # Configure conversion parameters
        rvc.configure({
            "f0up_key": 0,  # No pitch adjustment needed
            "index_rate": float(speaker_index),
            "protect": 0.33,
            "f0method": "rmvpe"
        })

        # Perform the conversion
        rvc.infer_file(input_path, output_path)

        return send_file(output_path, as_attachment=True, download_name="converted.wav", mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

@app.route("/models", methods=["GET"])
def list_models():
    """List all available models in the models directory"""
    models_dir = Path("models")
    models = [f.name for f in models_dir.glob("*.pth")]
    return jsonify({"models": models})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
