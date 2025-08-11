FROM mcr.microsoft.com/devcontainers/cpp:0-debian

# タイムゾーンとX11開発パッケージをインストール
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN apt-get update && \
    apt-get install -y tzdata libxext-dev libx11-dev && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*