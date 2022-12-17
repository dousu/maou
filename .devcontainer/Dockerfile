FROM mcr.microsoft.com/vscode/devcontainers/base:bullseye

# install bash-completion
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends bash-completion \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ARG USERNAME="vscode"

# install hadolint
USER ${USERNAME}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG HADOLINT_VERSION
RUN mkdir -p /home/${USERNAME}/.local/bin \
    && curl -sSL https://github.com/hadolint/hadolint/releases/download/${HADOLINT_VERSION}/hadolint-Linux-x86_64 -o /home/${USERNAME}/.local/bin/hadolint \
    && chmod +x /home/${USERNAME}/.local/bin/hadolint
