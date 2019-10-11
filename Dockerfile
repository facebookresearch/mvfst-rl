## Demonstration of running an experiment using the model inside the Pantheon environment.

# This Dockerfile can be used to run any of the schemes Pantheon has available
# by specifying the SCHEMENAME argument, inside a Docker container with high
# privileges. We are using it to demonstrate using a trained mvfst_rl model.

# In particular, this container is not used for training the model.

# 1) If using Docker Desktop on a Mac, it is a good idea to increase its memory limits
# because the default 2GB is too small to build mvfst.

# 2) Build the docker image using
#     docker build --tag mvfst_rl --build-arg SCHEMENAME=mvfst_rl - < Dockerfile
# where Dockerfile is this file.

# 3) Run the image using
#     CAPS='--cap-add=NET_ADMIN --cap-add=SYS_ADMIN'
#     sudo docker run --name c_mvfst_rl ${CAPS:?} --rm -t -i mvfst_rl
# 
# Inside the container, you can run any of the mvfst schemes because they all depend
# on the same setup. For example you can type
#     sudo -u runner -H src/experiments/test.py local --schemes mvfst_bbr --flows 1
# for the bbr scheme. The mvfst_rl scheme (running the trained model) can also be run with
#     . 1

FROM ubuntu:18.04

RUN echo Europe/London > /etc/timezone && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git \
        python-pip \
        python-yaml \
        python-matplotlib \
        sudo \
        ntp \
        ntpdate \
        mahimahi \
        autogen \
        debhelper autotools-dev dh-autoreconf iptables pkg-config iproute2 && \
    pip install tabulate && \
    useradd runner && \
    mkdir -m 777 ~runner && \
    chown runner: ~runner

RUN sudo -u runner -H git clone https://github.com/StanfordSNR/pantheon.git ~runner/pantheon

WORKDIR /home/runner/pantheon

RUN sudo -u runner -H git submodule update --init --recursive

RUN cd ~runner/pantheon/third_party/pantheon-tunnel && ./autogen.sh && \
    ./configure && make && make install

ARG SCHEMENAME

RUN src/experiments/setup.py --install-deps --schemes $SCHEMENAME

RUN src/experiments/setup.py --setup --schemes $SCHEMENAME

RUN echo 'mkdir -p /dev/net && mknod /dev/net/tun c 10 200'   > prelim.sh && \
    echo 'mount -o remount rw /proc/sys'                     >> prelim.sh && \
    echo 'chmod o+w tmp'                                     >> prelim.sh && \
    echo 'echo Please run \". 0\" or \". 1\" to run a test.' >> prelim.sh && \
    echo "sudo -u runner -H src/experiments/test.py local --schemes $SCHEMENAME --flows 1" > 1 && \
    echo "sudo -u runner -H src/experiments/test.py local --schemes $SCHEMENAME --flows 0" > 0

CMD bash --init-file /home/runner/pantheon/prelim.sh

