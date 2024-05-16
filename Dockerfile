FROM python:3.10

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir /app
WORKDIR /app

COPY ViT.py .
COPY server.py .

ENTRYPOINT ["python3", "server.py"]