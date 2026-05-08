FROM python:3.7

RUN pip install --upgrade --no-cache-dir pipenv

RUN pipenv --python 3.7

RUN pipenv install

