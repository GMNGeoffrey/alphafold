# A docker image to run alphafold on ROCm

# This is broken into two stages so that you can build just the deps stage as
# its own image and mount your own alphafold source directory instead of copying
# it in, e.g.
# docker build --target deps -t alphafold-deps - <docker/rocm.Dockerfile

ARG BASE_IMAGE=docker.io/rocm/jax:rocm7.2-jax0.8.0-py3.12
FROM ${BASE_IMAGE} AS deps

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV JAX_PLATFORMS="gpu,cpu"

# The paths in these base images are frequently broken, pointing to
# /opt/rocm/jax-7.2 instead of /opt/rocm/jax-7.2.0, for instance. So we set them
# ourselves. /opt/rocm should generally point to the correct thing regardless of
# version. See https://github.com/ROCm/rocm-jax/issues/124
ENV LLVM_PATH="/opt/rocm/llvm"
ENV HIP_PATH="/opt/rocm"
ENV ROCM_PATH="/opt/rocm"
ENV LD_LIBRARY_PATH="/opt/rocm/lib"
ENV PATH="/opt/rocm/opencl/bin:/opt/rocm/hcc/bin:/opt/rocm/llvm/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update --quiet \
  && apt-get install --no-install-recommends --yes --quiet \
    build-essential \
    cmake \
    git \
    hmmer \
    kalign \
    tzdata \
    # Workaround for https://github.com/ROCm/rocm-jax/issues/163
    libdw1t64 \
    wget

RUN git clone --branch v3.3.0 --single-branch https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
  && pushd /tmp/hh-suite \
  # missing include for cstdint.h otherwise breaks the build
  && wget -O - https://github.com/soedinglab/hh-suite/commit/bf3f42747ca4ddbd72e1c142aa3b648793c59045.patch | patch -p1 \
  && mkdir build \
  && cd build \
  && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
  && make -j && make install \
  && ln -s /opt/hhsuite/bin/* /usr/bin \
  && popd \
  && rm -rf /tmp/hh-suite

# Mostly the same as the upstream requirements.txt, but updated for
# compatibility with the base image JAX and Python versions.
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install \
    'absl-py==1.0.0' \
    'biopython==1.85' \
    'dm-haiku==0.0.15' \
    'ml-collections==0.1.0' \
    'numpy==2.4.1' \
    'pytest<8.5.0' \
    'setuptools<72.0.0' \
    'tensorflow-cpu==2.20.0' \
    # Someone republished this on PyPI, so we don't have to go through conda, hence the odd name
    'pdbfixer-wheel==1.11.0' \
    'openmm==8.3.1'

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

FROM deps

COPY --link . /app/alphafold
# I assume this tiny text file isn't just included upstream because of its GPL license
ADD --link \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt \
  /app/alphafold/alphafold/common/

WORKDIR /app/alphafold
