FROM python:3.10.16-slim-bullseye

# dependencies 
RUN apt update && apt install -y build-essential git libgl1 libglib2.0-0 curl

# install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install -y git-lfs
# env
ENV OMP_NUM_THREADS=18

# pip requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip ipykernel
RUN pip install --no-cache-dir -r /app/requirements.txt

# set workdir
WORKDIR /app