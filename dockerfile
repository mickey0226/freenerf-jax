FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 常に非対話モードで実行する設定
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    # PPAの追加やpipインストールに必要な基本パッケージをインストール
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    screen \
    wget && \
    # Python 3.7を提供しているdeadsnakes PPAを追加
    add-apt-repository -y ppa:deadsnakes/ppa && \
    # PPA追加後に再度パッケージリストを更新
    apt-get update && \
    # 必要な全パッケージを一度にインストール
    # (python3.7-distutilsはget-pip.pyの実行に必要)
    apt-get install -y --no-install-recommends \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    git \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    # pipをインストール
    curl -sS https://bootstrap.pypa.io/pip/3.7/get-pip.py | python3.7 && \
    # デフォルトのpython3コマンドをpython3.7に向ける
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    # 後片付けを行い、イメージサイズを削減
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    tensorflow \
    tensorflow-probability[tf] \
    # JAXのバージョンを0.3.15に固定
    jax==0.3.15 \
    jaxlib==0.3.15+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    # その他のライブラリ
    flax==0.5.1 \
    optax \
    gin-config \
    absl-py \
    opencv-python \
    scikit-image==0.17.2 \
    tqdm \
    wandb \
    lpips \
    mediapy \
    dm-pix \
    oryx

# 作業ディレクトリの設定
WORKDIR /root

# コンテナ起動時のデフォルトコマンド
CMD ["bash"]