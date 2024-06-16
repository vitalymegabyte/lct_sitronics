FROM ubuntu:focal

WORKDIR /work
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get install -y g++ gdal-bin libgdal-dev python3-pip python3-gdal libopencv-dev

COPY requirements.txt .
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal &&\
    export C_INCLUDE_PATH=/usr/include/gdal &&\
    pip install -r requirements.txt

COPY . .
