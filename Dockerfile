# https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/running.html
FROM nvcr.io/nvidia/jax:24.10-py3

# Create User
ARG UID
ARG MYUSER
RUN useradd -u $UID --create-home ${MYUSER}
USER ${MYUSER}

# Set default workdir
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} --chmod=765 . .

USER root

# Intall tmux
RUN apt-get update && \
    apt-get install -y tmux

# Install remaining packages.
RUN pip install -e .

USER ${MYUSER}

# Set environment variables.
RUN export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

# Secrets and Debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""
RUN git config --global --add safe.directory /home/${MYUSER}