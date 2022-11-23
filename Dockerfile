FROM nvidia/cuda:11.5.0-devel-ubuntu20.04
RUN apt-get update
RUN apt-get install -y python3 pip wget unzip
RUN pip3 install numpy pandas matplotlib
RUN apt-get install -y cuda-nsight-systems-11-5 --no-install-recommends

WORKDIR "/LightTraffic"
