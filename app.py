from flask import Flask, request, send_file, jsonify
from pathlib import Path
import os
import tempfile
from dotenv import load_dotenv
from rvc_python.infer import RVCInference

# curl -X POST \
#   -F "file=@konstantin.wav" \
#   -F "speaker_index=SPEAKER_01" \
#   -F "gender=female" \
#   http://localhost:5000/convert \
#   --output converted.wav
#   -v

load_dotenv()

app = Flask(__name__)

# Initialize RVC with CUDA if available
# device = "cuda:0" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"
device = "cuda:0"
rvc = RVCInference(device=device)

# Define available models for different genders
AVAILABLE_MODELS = {
    "male": {
        "SPEAKER_01": "rvc_models/male_1/model.pth",
        "SPEAKER_02": "rvc_models/male_2/model.pth",
        "SPEAKER_03": "rvc_models/male_3/model.pth"
    },
    "female": {
        "SPEAKER_01": "rvc_models/female_1/model.pth",
        "SPEAKER_02": "rvc_models/female_2/model.pth",
        # "SPEAKER_03": "rvc_models/female_3/model.pth"
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
    speaker_index = request.form.get("speaker_index", "SPEAKER_01")
    gender = request.form.get("gender", "male").lower()

    # Set pitch based on gender
    pitch_adjust = 3 if gender == "male" else -3

    # Select appropriate model based on gender and speaker index
    if gender not in AVAILABLE_MODELS or speaker_index not in AVAILABLE_MODELS[gender]:
        return jsonify({"error": "Invalid gender or speaker index"}), 400

    model_path = AVAILABLE_MODELS[gender][speaker_index]
    print(f"Loading model from: {model_path}")  # Debug log
    rvc.load_model(model_path)

    # Set RVC parameters
    rvc.f0up_key = pitch_adjust
    rvc.f0method = "rmvpe"
    rvc.index_rate = float(speaker_index.split('_')[1])
    rvc.protect = 0.33

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
        input_path = temp_in.name
        file.save(input_path)
        
        # Debug: Check input file size
        input_size = os.path.getsize(input_path)
        print(f"Input file size: {input_size} bytes")
        if input_size < 1000:
            return jsonify({"error": "Input file too small or empty"}), 400

    try:
        output_path = input_path + "_converted.wav"
        
        # Perform the conversion
        print(f"Converting with parameters: f0up_key={rvc.f0up_key}, index_rate={rvc.index_rate}")
        rvc.infer_file(input_path, output_path)

        # Check output file
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            return jsonify({"error": "Conversion failed - output file is empty or too small"}), 500

        return send_file(output_path, as_attachment=True, download_name="converted.wav", mimetype="audio/wav")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")  # Debug log
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
