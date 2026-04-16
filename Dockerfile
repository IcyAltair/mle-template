FROM python:3.14-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt