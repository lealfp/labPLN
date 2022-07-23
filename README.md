A repository to execute Natural Language Processing experiments.


We provide two indepentent docker containers to locally run the `PLN Laboratory` environment. 



> The container guarantee the required minimal configuration to run the code. Read [docker](https://docs.docker.com/install/) and [docker-compose](https://docs.docker.com/compose/install/) documentations to install docker.

> In order to execute `docker` without `sudo`, read this link: https://docs.docker.com/engine/install/linux-postinstall/.


# 1. Jupyter Container 
## Instructions (for Linux users)

Clone `labPLN` repository, get into it, checkout `master` branch, and build the docker image:

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

> You must change the `[TOKEN]` keyword by the actual token displayed on terminal by the server.

# 2. GPU Container

```bash
 docker run --gpus all -it -v ~/git/labPLN:/app tensorflow/tensorflow:latest-gpu
```
