FROM python:3.10.16-slim-bullseye

# dependencies 
RUN apt update && apt install -y build-essential git libgl1 libglib2.0-0

# env
ENV OMP_NUM_THREADS=18

# pip requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip ipykernel
RUN pip install --no-cache-dir -r /app/requirements.txt

# set workdir
WORKDIR /app