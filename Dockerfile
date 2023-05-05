# FROM ubuntu:20.04
FROM python:3.10.5-slim
ARG DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y python3.10

# RUN rm -rf /workspace/*
WORKDIR /home
COPY . .
RUN pip install --upgrade pip
# ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
# ADD . .