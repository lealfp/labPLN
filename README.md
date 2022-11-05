# Summary
Repository to execute Natural Language Processing experiments. It provides two independent environments built as docker containers:
- Jupyter notebooks: the environment used to run Topic Attention, as described in the paper.
- GPU-enabled Python container: an environment used to train the deep neural networks analysed on the paper.

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

Once the process is done, you can access the notebook through http://127.0.0.1:8888/lab?token=[TOKEN]. This environment provides a jupyter notebook interface .

> You must change the `[TOKEN]` keyword by the actual token displayed on the terminal

> If you want to run the code in the `src` folder, then access the GPU-enabled container:
> ```bash
> docker run --gpus all -it -v ~/git/labPLN:/app tensorflow/tensorflow:latest-gpu
> ```
> From here, you must follow the instructions in README.md associated to the experiment. 
