FROM mcr.microsoft.com/vscode/devcontainers/base:bullseye

# install python3
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3 python3-distutils \
    # clean up
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ARG USERNAME="vscode"

# install poetry
# https://python-poetry.org/docs/#installing-with-the-official-installer
USER ${USERNAME}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && echo "export PATH=\$PATH:/home/vscode/.local/bin" >> /home/${USERNAME}/.bash_profile

# install hadolint
ARG HADOLINT_VERSION
RUN curl -sSL https://github.com/hadolint/hadolint/releases/download/${HADOLINT_VERSION}/hadolint-Linux-x86_64 -o /home/${USERNAME}/.local/bin/hadolint \
    && chmod +x /home/${USERNAME}/.local/bin/hadolint
