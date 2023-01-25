FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update
RUN apt-get -y install python3-pip vim git
RUN apt-get -y install libfreetype-dev libfreetype6 libfreetype6-dev

RUN pip install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install fastapi pandas requests torch fsspec && pip install "uvicorn[standard]"

RUN mkdir /RE_module && mkdir /RE_module/src && mkdir /RE_module/config && mkdir /RE_module/models && mkdir /RE_module/data
COPY spanREL/src /RE_module/src
WORKDIR /RE_module/src

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

CMD ["/bin/bash"]
