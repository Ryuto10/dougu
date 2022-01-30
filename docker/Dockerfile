FROM python:3.9.7

USER root

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \
    | POETRY_HOME=/opt/poetry python \
    && cd /usr/local/bin \
    && ln -s /opt/poetry/bin/poetry \
    && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip \
    && poetry install \
    && rm pyproject.toml poetry.lock \
