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

RUN sudo -u runner -H src/experiments/setup.py --setup --schemes $SCHEMENAME

RUN echo 'mkdir -p /dev/net && mknod /dev/net/tun c 10 200'   > prelim.sh && \
    echo 'mount -o remount rw /proc/sys'                     >> prelim.sh && \
    echo 'chmod o+w tmp'                                     >> prelim.sh && \
    echo 'echo Please run \". 0\" or \". 1\" to run a test.' >> prelim.sh && \
    echo "sudo -u runner -H src/experiments/test.py local --schemes $SCHEMENAME --flows 1" > 1 && \
    echo "sudo -u runner -H src/experiments/test.py local --schemes $SCHEMENAME --flows 0" > 0

CMD bash --init-file /home/runner/pantheon/prelim.sh

