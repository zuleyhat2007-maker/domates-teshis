FROM python:3.11-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user PYTHONUNBUFFERED=1 PORT=7860 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

# Domates filtresinin ImageNet ağırlıklarını kurulum sırasında indir (ilk açılış hızlansın)
RUN python -c "from tensorflow.keras.applications import MobileNetV2; MobileNetV2(include_top=False, pooling='avg', input_shape=(224,224,3), weights='imagenet')"

EXPOSE 7860
CMD ["python", "app.py"]
