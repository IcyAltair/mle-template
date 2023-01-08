FROM python:3.10

ENV PYTHON UNBUFFERED 1

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

