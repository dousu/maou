FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310

COPY . /app
WORKDIR /app

RUN pip install poetry && poetry sync --without dev

CMD ["poetry", "run", "maou"]
