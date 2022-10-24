Repository to execute Natural Language Processing experiments. It provides two independent environments built as docker containers:
- Jupyter notebooks
- GPU-enabled Python container

# Linux instructions to run

To access the containers, clone `labPLN` repository, get into it, checkout `master` branch, and build the docker image:

```bash
cd ~/git
git clone https://github.com/lealfp/labPLN.git
cd labPLN
git pull origin master
docker build . -t labPLN
```

Then, start up the docker container:

```bash
docker-compose up
```

Once the process is done, you can access the notebook through http://127.0.0.1:8888/lab?token=[TOKEN].

> You must change the `[TOKEN]` keyword by the actual token displayed on the terminal

To access the GPU container:

```bash
 docker run --gpus all -it -v ~/git/labPLN:/app tensorflow/tensorflow:latest-gpu
```
