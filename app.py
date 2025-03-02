from flask import Flask, request, send_file, jsonify
from pathlib import Path
import os
import tempfile
from dotenv import load_dotenv
import torch  # Для проверки CUDA
from rvc_python.infer import RVCInference
import logging
import zipfile

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env
load_dotenv()

# Инициализация Flask-приложения
app = Flask(__name__)

# Проверка доступности CUDA и выбор устройства
device = "cuda:0" if os.getenv("USE_CUDA", "false").lower() == "true" and torch.cuda.is_available() else "cpu"
logger.info(f"Выбрано устройство: {device}")

# Инициализация RVC с выбранным устройством
rvc = RVCInference(device=device)
torch.backends.cudnn.benchmark = True

# Определение доступных моделей для разных полов
AVAILABLE_MODELS = {
    "male": {
        "SPEAKER_02": "rvc_models/male_1/model.pth",
        "SPEAKER_01": "rvc_models/male_2/model.pth",
        "SPEAKER_03": "rvc_models/male_3/model.pth"
    },
    "female": {
        "SPEAKER_01": "rvc_models/female_1/model.pth",
        "SPEAKER_02": "rvc_models/female_2/model.pth",
        # "SPEAKER_03": "rvc_models/female_3/model.pth"  # Закомментировано, если модель отсутствует
    }
}

# Предзагрузка всех моделей при старте приложения
loaded_models = {}
for gender, speakers in AVAILABLE_MODELS.items():
    for speaker, model_path in speakers.items():
        if os.path.exists(model_path):
            try:
                rvc.load_model(model_path)
                loaded_models[(gender, speaker)] = model_path
                logger.info(f"Загружена модель для {gender} {speaker} из {model_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели для {gender} {speaker}: {str(e)}")
        else:
            logger.warning(f"Файл модели не найден: {model_path}")

@app.route("/convert_batch", methods=["POST"])
def convert_audio_batch():
    """Конвертация пакета аудиофайлов с использованием RVC."""
    # Проверка наличия файлов в запросе
    if "files" not in request.files:
        return jsonify({"error": "Файлы не предоставлены"}), 400
    
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "Файлы не выбраны"}), 400
    
    # Получение общих параметров для всех файлов
    speaker_index = request.form.get("speaker_index", "SPEAKER_01")
    gender = request.form.get("gender", "male").lower()
    
    # Проверка модели
    if (gender, speaker_index) not in loaded_models:
        return jsonify({"error": "Неверный пол или индекс спикера"}), 400
    
    # Установка параметров RVC
    pitch_adjust = 0 if gender == "male" else 2
    rvc.f0up_key = pitch_adjust
    rvc.f0method = "rmvpe"
    rvc.index_rate = float(speaker_index.split('_')[1])
    rvc.protect = 0.33
    
    # Создаем временную директорию для результатов
    with tempfile.TemporaryDirectory() as temp_dir:
        input_files = []
        output_files = []
        
        # Сохраняем все входные файлы
        for i, file in enumerate(files):
            input_path = os.path.join(temp_dir, f"input_{i}.wav")
            output_path = os.path.join(temp_dir, f"output_{i}.wav")
            file.save(input_path)
            input_files.append(input_path)
            output_files.append(output_path)
        
        try:
            # Обработка файлов (можно распараллелить на уровне PyTorch, если модель это поддерживает)
            for input_path, output_path in zip(input_files, output_files):
                rvc.infer_file(input_path, output_path)
            
            # Упаковка результатов (например, в архив)
            result_zip = os.path.join(temp_dir, "results.zip")
            with zipfile.ZipFile(result_zip, 'w') as zipf:
                for output_path in output_files:
                    if os.path.exists(output_path):
                        zipf.write(output_path, os.path.basename(output_path))
            
            return send_file(result_zip, as_attachment=True, download_name="converted_batch.zip")
            
        except Exception as e:
            logger.error(f"Ошибка во время пакетной конвертации: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route("/convert", methods=["POST"])
def convert_audio():
    """Конвертация аудиофайла с использованием RVC."""
    # Проверка наличия файла в запросе
    if "file" not in request.files:
        return jsonify({"error": "Файл не предоставлен"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400

    # Получение параметров из формы
    speaker_index = request.form.get("speaker_index", "SPEAKER_01")
    gender = request.form.get("gender", "male").lower()

    # Проверка, загружена ли модель для указанного пола и спикера
    if (gender, speaker_index) not in loaded_models:
        return jsonify({"error": "Неверный пол или индекс спикера"}), 400

    # Установка сдвига высоты тона в зависимости от пола
    pitch_adjust = 0 if gender == "male" else 2

    # Установка параметров RVC
    rvc.f0up_key = pitch_adjust
    rvc.f0method = "rmvpe"
    rvc.index_rate = float(speaker_index.split('_')[1])  # Предполагается формат "SPEAKER_01"
    rvc.protect = 0.33

    # Сохранение входного файла во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
        input_path = temp_in.name
        file.save(input_path)

        # Проверка размера входного файла
        input_size = os.path.getsize(input_path)
        logger.info(f"Размер входного файла: {input_size} байт")
        if input_size < 1000:
            os.remove(input_path)
            return jsonify({"error": "Входной файл слишком маленький или пустой"}), 400

    try:
        output_path = input_path + "_converted.wav"

        # Выполнение конвертации
        logger.info(f"Конвертация с параметрами: f0up_key={rvc.f0up_key}, index_rate={rvc.index_rate}")
        rvc.infer_file(input_path, output_path)

        # Проверка выходного файла
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            logger.error("Конвертация не удалась - выходной файл пустой или слишком маленький")
            return jsonify({"error": "Конвертация не удалась - выходной файл пустой или слишком маленький"}), 500

        # Отправка сконвертированного файла
        return send_file(output_path, as_attachment=True, download_name="converted.wav", mimetype="audio/wav")

    except Exception as e:
        logger.error(f"Ошибка во время конвертации: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Очистка временных файлов
        if os.path.exists(input_path):
            os.remove(input_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)

@app.route("/models", methods=["GET"])
def list_models():
    """Список всех доступных моделей в директории models."""
    models_dir = Path("models")
    models = [f.name for f in models_dir.glob("*.pth")]
    return jsonify({"models": models})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)