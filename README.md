# 1. Summary

Repository containing the source code to reproduce the ABT and LDA experiments reported in the paper "Semantic Representations based on Neural Topic Models" and the PhD thesis "Semantic Representations based on Language Models":

```bibtext
@article{pantoja2026abt,
  title     = {Semantic Representations based on Neural Topic Models},
  author    = {Pantoja, F. and Santanchè, A. and Medeiros, C.},
  journal   = {Journal of Universal Computer Science},
  year      = {2026}
}
```

```bibtext
@phdthesis{pantoja2025thesis,
  title   = {Semantic Representations based on Language Models},
  author  = {Pantoja, F.},
  school  = {Universidade Estadual de Campinas},
  year    = {2025}
}
```
 
This project provides a Jupyter environment packaged as a Docker container ready to run the methods. Jupyter enables editing/running the notebooks directly in the browser.


> Docker containers are used to guarantee isolated environments with the required minimal configuration to run the code. Read [docker](https://docs.docker.com/install/) e [docker-compose](https://docs.docker.com/compose/install/) documentations to install docker.

# 2. Linux instructions to run

Open the Linux terminal using <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd>.

Navigate to the folder you downloaded/extracted and build the Docker image using the following commands:

```bash
cd labPLN
docker build . -t labpln
``` 

Then, start up the container:

```bash
docker compose up
```

Once the process is complete, you can access the notebook at: 

http://127.0.0.1:8888/lab?token=[TOKEN]


> You must change the `[TOKEN]` keyword by the actual token displayed on the terminal

You can navigate through the `notebooks/topic_modeling` directory in the Jupyter interface to access the notebooks for 

- ABT: Code to reproduce the Attention-based Topics model.
- LDA: Code to reproduce the Latent Dirichlet Allocation  model
- ABT[minimal]: a reduced version of ABT (without figure plots) provided to check the computational costs

# 3. Input description

The environment includes the MovieLens dataset. The input is the `movies.csv` file, which contains a column named `title` to register short sentences.

One can add additional datasets to the `input` folder. 

The CliCR dataset analyzed in the paper must be requested from the dataset authors (https://github.com/clips/clicr). The input file used in this project is `train1.0.json` file, which includes the `title` field along with other attributes describing each clinical case.
