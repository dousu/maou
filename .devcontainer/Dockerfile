FROM mcr.microsoft.com/vscode/devcontainers/base:bullseye

# install python3
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3 python3-distutils \
    # clean up
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# install poetry
# https://python-poetry.org/docs/#installing-with-the-official-installer
USER vscode
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && echo 'export PATH=$PATH:/home/vscode/.local/bin' >> /home/vscode/.bash_profile
