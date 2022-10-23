Repository to execute Natural Language Processing experiments, providing two independent docker containers:

# 1. Jupyter Container
## Instructions (for Linux users)

To access the jupyter notebooks, clone `labPLN` repository, get into it, checkout `master` branch, and build the docker image:

```bash
git clone https://github.com/lealfp/labPLN.git
cd labPLN
git pull origin master

docker build . -t labPLN
```

Then, start up the docker container:

```bash
docker-compose up
```

Once the process is done, you can access http://127.0.0.1:8888/lab?token=[TOKEN].

> You must change the `[TOKEN]` keyword by the actual token displayed on the terminal

# 2. GPU-enabled Container

To access the GPU container:

```bash
 docker run --gpus all -it -v ~/git/labPLN:/app tensorflow/tensorflow:latest-gpu
```
