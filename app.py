from flask import Flask, request, send_file, jsonify
from pathlib import Path
import os
import tempfile
from dotenv import load_dotenv
import torch  # Для проверки CUDA
from rvc_python.infer import RVCInference
import logging
import zipfile
import concurrent.futures
import time
import gc  # Для принудительной сборки мусора

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env
load_dotenv()

# Инициализация Flask-приложения
app = Flask(__name__)

# Проверка доступности CUDA и выбор устройства
device = "cuda:0" if os.getenv("USE_CUDA", "true").lower() == "true" and torch.cuda.is_available() else "cpu"
logger.info(f"Выбрано устройство: {device}")

# Оптимизация CUDA для PyTorch
if device.startswith("cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    
    # Настройка управления памятью CUDA для избежания фрагментации
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Установка ограничения на кеш памяти
    torch.cuda.set_per_process_memory_fraction(0.8)  # Использовать не более 80% GPU памяти
    
    logger.info("Включены оптимизации CUDA TF32 и управление памятью")

# Определение доступных моделей для разных языков и полов
RU_AVAILABLE_MODELS = {
    "male": {
        "SPEAKER_01": "rvc_models/male_2/model.pth",
        "SPEAKER_02": "rvc_models/male_1/model.pth",
        "SPEAKER_03": "rvc_models/male_3/model.pth"
    },
    "female": {
        "SPEAKER_01": "rvc_models/female_1/model.pth",
        "SPEAKER_02": "rvc_models/female_2/model.pth",
    }
}

ES_AVAILABLE_MODELS = {
    "male": {
        "SPEAKER_01": "rvc_models/es_male_2/model.pth",
        "SPEAKER_02": "rvc_models/es_male_1/model.pth",
    },
    "female": {
        "SPEAKER_01": "rvc_models/es_female_1/model.pth",
    }
}

HI_AVAILABLE_MODELS = {
    "male": {
        "SPEAKER_01": "rvc_models/hi_male_1/model.pth",
        "SPEAKER_02": "rvc_models/hi_male_2/model.pth",
        "SPEAKER_03": "rvc_models/hi_male_3/model.pth",
    },
    "female": {
        "SPEAKER_01": "rvc_models/hi_female_1/model.pth",
    }
}

# Словарь для выбора моделей по языку
LANGUAGE_MODELS = {
    "ru": RU_AVAILABLE_MODELS,
    "es": ES_AVAILABLE_MODELS,
    "hi": HI_AVAILABLE_MODELS
}

# Создаем отдельные экземпляры RVC для каждой модели
rvc_instances = {}

# Предзагружаем модели в отдельные экземпляры RVC для всех языков
for language, language_models in LANGUAGE_MODELS.items():
    for gender, speakers in language_models.items():
        for speaker, model_path in speakers.items():
            if os.path.exists(model_path):
                try:
                    # Создаем новый экземпляр для каждой модели
                    rvc_instance = RVCInference(device=device)
                    rvc_instance.load_model(model_path)
                    
                    # Установка оптимальных параметров для данной модели
                    pitch_adjust = 0 if gender == "male" else 2
                    rvc_instance.f0up_key = pitch_adjust
                    rvc_instance.f0method = "rmvpe"  # или другой быстрый метод
                    rvc_instance.index_rate = float(speaker.split('_')[1])
                    rvc_instance.protect = 0.33
                    
                    rvc_instances[(language, gender, speaker)] = rvc_instance
                    logger.info(f"Загружена и настроена модель для {language} {gender} {speaker} из {model_path}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки модели для {language} {gender} {speaker}: {str(e)}")
            else:
                logger.warning(f"Файл модели не найден: {model_path}")

# Устанавливаем резервный экземпляр на случай отсутствия модели
default_rvc = RVCInference(device=device)

def process_file(input_path, output_path, language, gender, speaker_index):
    """Функция для обработки одного файла в отдельном потоке"""
    try:
        start_time = time.time()
        
        # Очистка кеша CUDA перед обработкой
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            gc.collect()
        
        # Получаем соответствующий экземпляр RVC
        rvc_instance = rvc_instances.get((language, gender, speaker_index), default_rvc)
        if (language, gender, speaker_index) not in rvc_instances:
            logger.warning(f"Используется запасной экземпляр для {language} {gender} {speaker_index}")
            
            # Устанавливаем параметры, если используется запасной экземпляр
            pitch_adjust = 0 if gender == "male" else 2
            default_rvc.f0up_key = pitch_adjust
            default_rvc.f0method = "rmvpe"
            default_rvc.index_rate = float(speaker_index.split('_')[1])
            default_rvc.protect = 0.33
        
        # Проверяем, существует ли входной файл и не пуст ли он
        if not os.path.exists(input_path) or os.path.getsize(input_path) < 500:
            logger.warning(f"Пропуск обработки файла {input_path}: файл не существует или слишком маленький")
            return False
            
        # Выполняем конвертацию с обработкой ошибок памяти
        try:
            rvc_instance.infer_file(input_path, output_path)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory при обработке {input_path}: {str(e)}")
            # Очищаем кеш и пробуем еще раз
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)  # Даем время на очистку памяти
            
            try:
                logger.info(f"Повторная попытка обработки {input_path} после очистки памяти")
                rvc_instance.infer_file(input_path, output_path)
            except Exception as retry_e:
                logger.error(f"Повторная попытка не удалась для {input_path}: {str(retry_e)}")
                return False
        except Exception as e:
            logger.error(f"Общая ошибка при обработке {input_path}: {str(e)}")
            return False
        
        # Очистка кеша CUDA после обработки
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            gc.collect()
        
        # Проверяем результат
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            elapsed = time.time() - start_time
            logger.info(f"Файл {os.path.basename(input_path)} обработан за {elapsed:.2f} сек")
            return True
        else:
            logger.error(f"Ошибка при обработке {input_path}: выходной файл отсутствует или слишком мал")
            return False
    except Exception as e:
        logger.error(f"Исключение при обработке {input_path}: {str(e)}")
        return False
    finally:
        # Принудительная очистка памяти в конце
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

@app.route("/convert_batch", methods=["POST"])
def convert_audio_batch():
    """Конвертация пакета аудиофайлов с использованием RVC."""
    start_time = time.time()
    
    # Проверка наличия файлов в запросе
    if "files" not in request.files:
        return jsonify({"error": "Файлы не предоставлены"}), 400
    
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "Файлы не выбраны"}), 400
    
    # Получение общих параметров для всех файлов
    speaker_index = request.form.get("speaker_index", "SPEAKER_01")
    gender = request.form.get("gender", "male").lower()
    language = request.form.get("language", "ru").lower()
    
    # Проверка языка
    if language not in LANGUAGE_MODELS:
        return jsonify({"error": f"Неверный язык: {language}. Доступны: ru, es, hi"}), 400
    
    # Проверка модели
    if (language, gender, speaker_index) not in rvc_instances:
        # Использовать резервную модель, если указанная не найдена
        logger.warning(f"Модель для {language} {gender} {speaker_index} не найдена, поиск альтернативы")
        
        # Попытка найти другую модель для того же языка и пола
        found = False
        for (lang, g, s) in rvc_instances.keys():
            if lang == language and g == gender:
                speaker_index = s
                logger.info(f"Используется альтернативная модель: {language} {gender} {speaker_index}")
                found = True
                break
                
        if not found:
            return jsonify({"error": f"Неверный пол или индекс спикера для языка {language}"}), 400
    
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
        
        logger.info(f"Начало обработки {len(input_files)} файлов для {language} {gender} {speaker_index}")
        
        try:
            # Параллельная обработка файлов с использованием ThreadPoolExecutor
            # Устанавливаем max_workers на optimal, чтобы управлять потоками эффективно
            # Для GPU-операций используем меньше потоков для избежания CUDA OOM
            if device.startswith("cuda"):
                max_workers = min(2, len(input_files))  # только 2 потока для GPU
            else:
                max_workers = min(4, len(input_files))  # больше потоков для CPU
            
            success_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Создаем задачи для обработки файлов
                future_to_file = {
                    executor.submit(process_file, input_path, output_path, language, gender, speaker_index): 
                    (i, input_path, output_path) 
                    for i, (input_path, output_path) in enumerate(zip(input_files, output_files))
                }
                
                # Обрабатываем результаты по мере завершения задач
                for future in concurrent.futures.as_completed(future_to_file):
                    i, input_path, output_path = future_to_file[future]
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Ошибка при обработке файла {i}: {str(e)}")
            
            # Упаковка результатов в архив
            result_zip = os.path.join(temp_dir, "results.zip")
            with zipfile.ZipFile(result_zip, 'w') as zipf:
                for output_path in output_files:
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        zipf.write(output_path, os.path.basename(output_path))
            
            total_time = time.time() - start_time
            logger.info(f"Обработка завершена: {success_count}/{len(input_files)} файлов за {total_time:.2f} сек ({total_time/len(input_files):.2f} сек/файл)")
            
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
    language = request.form.get("language", "ru").lower()

    # Проверка языка
    if language not in LANGUAGE_MODELS:
        return jsonify({"error": f"Неверный язык: {language}. Доступны: ru, es, hi"}), 400

    # Проверка, загружена ли модель для указанного языка, пола и спикера
    if (language, gender, speaker_index) not in rvc_instances:
        # Использовать резервную модель, если указанная не найдена
        logger.warning(f"Модель для {language} {gender} {speaker_index} не найдена, поиск альтернативы")
        
        # Попытка найти другую модель для того же языка и пола
        found = False
        for (lang, g, s) in rvc_instances.keys():
            if lang == language and g == gender:
                speaker_index = s
                logger.info(f"Используется альтернативная модель: {language} {gender} {speaker_index}")
                found = True
                break
                
        if not found:
            return jsonify({"error": f"Неверный пол или индекс спикера для языка {language}"}), 400

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

        # Очистка кеша CUDA перед обработкой
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            gc.collect()

        # Выполнение конвертации с использованием соответствующего экземпляра
        rvc_instance = rvc_instances.get((language, gender, speaker_index), default_rvc)
        
        start_time = time.time()
        logger.info(f"Начало конвертации файла {os.path.basename(input_path)} для {language}")
        
        # Обработка файла
        process_file(input_path, output_path, language, gender, speaker_index)
        
        elapsed = time.time() - start_time
        logger.info(f"Конвертация завершена за {elapsed:.2f} сек")

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
    """Список всех доступных моделей."""
    models = {}
    for (language, gender, speaker), instance in rvc_instances.items():
        if language not in models:
            models[language] = {}
        if gender not in models[language]:
            models[language][gender] = []
        models[language][gender].append(speaker)
    
    return jsonify(models)

@app.route("/status", methods=["GET"])
def get_status():
    """Возвращает информацию о состоянии сервера."""
    cuda_info = {}
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free_memory = total_memory - allocated
        
        cuda_info = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "total_memory": f"{total_memory:.2f} GB",
            "memory_allocated": f"{allocated:.2f} GB",
            "memory_reserved": f"{reserved:.2f} GB",
            "memory_free": f"{free_memory:.2f} GB",
            "memory_usage_percent": f"{(allocated/total_memory)*100:.1f}%",
            "max_memory_allocated": f"{torch.cuda.max_memory_allocated(0)/1024**3:.2f} GB",
        }
    
    return jsonify({
        "status": "running",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_info": cuda_info,
        "loaded_models_count": len(rvc_instances),
        "loaded_models": [f"{language}_{gender}_{speaker}" for language, gender, speaker in rvc_instances.keys()]
    })

if __name__ == "__main__":
    # Запуск с несколькими рабочими процессами для повышения производительности
    from waitress import serve
    threads = 4 if device == "cpu" else 2  # Меньше потоков для GPU, чтобы избежать конкуренции
    logger.info(f"Запуск сервера на http://0.0.0.0:5000 с {threads} потоками")
    serve(app, host="0.0.0.0", port=5000, threads=threads)