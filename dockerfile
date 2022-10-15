# syntax=docker/dockerfile:1
FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
