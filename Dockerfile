# Start with a base image containing Python runtime
FROM python:3.10

RUN apt-get update

# Install necessary packages 
RUN apt-get install -y tar gzip wget

# Set the working directory in the container
WORKDIR /root

# Clone the vits repo
RUN git clone https://github.com/jaywalnut310/vits.git
RUN mv /root/vits/* /root/ && \
    rm -rf /root/vits

# Install the necessary python libraries
RUN pip install Cython==0.29.21
RUN pip install librosa==0.8.0
RUN pip install phonemizer==2.2.1
RUN pip install scipy
RUN pip install numpy
RUN pip install torch
RUN pip install torchvision
RUN pip install matplotlib
RUN pip install Unidecode==1.1.1
RUN pip install pymupdf==1.22.3

# Setup monotonic_align
WORKDIR /root/monotonic_align
RUN mkdir monotonic_align
RUN python3 setup.py build_ext --inplace

# Set the working directory back to /vits
WORKDIR /root
RUN wget https://dl.fbaipublicfiles.com/mms/tts/eng.tar.gz -O eng.tar.gz
RUN tar -xzvf eng.tar.gz --no-same-owner
