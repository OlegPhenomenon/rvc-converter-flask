from flask import Flask, request, send_file, jsonify
from pathlib import Path
from scipy.io import wavfile
from rvc.modules.vc.modules import VC
import os
import tempfile

# # Настройка переменных окружения
# os.environ["index_root"] = "/content/models"
# os.environ["hubert_path"] = "/content/models/hubert_base.pt"
# os.environ["weight_root"] = "/content/models"
# os.environ["rmvpe_root"] = "/content/models"
# os.environ["weight_uvr5_root"] = "/content/models"
# os.environ["save_uvr_path"] = "/content/models"
# os.environ["TEMP"] = "/content/temp"
# os.environ["pretrained"] = "/content/models/hubert_base.pth"

# Инициализация Flask
app = Flask(__name__)

# Инициализация модели
vc = VC()
vc.get_vc("/content/models/kow.pth")

@app.route("/convert", methods=["POST"])
def convert_audio():
    # Проверяем, есть ли файл в запросе
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Сохраняем загруженный файл во временную директорию
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
        input_path = temp_in.name
        file.save(input_path)

    try:
        # Выполняем конвертацию
        tgt_sr, audio_opt, _, _ = vc.vc_single(1, Path(input_path))

        # Сохраняем результат во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            output_path = temp_out.name
            wavfile.write(output_path, tgt_sr, audio_opt)

        # Отправляем файл обратно клиенту
        return send_file(output_path, as_attachment=True, download_name="converted.wav", mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Удаляем временные файлы
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

# Запуск приложения
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
