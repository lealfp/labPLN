FROM python:3.7

#ENV PIN_PIPENV_VERSION=2021.5.29

RUN pip install --upgrade --no-cache-dir pipenv

WORKDIR /app
COPY ./src/ .
# COPY Pipfile Pipfile.lock bootstrap.sh ./

RUN pipenv --python 3.7
RUN pipenv install

## Start app
#EXPOSE 10040
#ENTRYPOINT ["./bootstrap.sh"]
