FROM python:3.8-buster

WORKDIR /app

COPY requirements.txt .

# Installer les dépendances avec un miroir alternatif et un délai d'attente plus long
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

RUN mkdir -p /app/Test_Music

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
