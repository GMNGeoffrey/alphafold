FROM docker.io/rocm/jax:rocm7.1-jax0.7.1-py3.12

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV JAX_PLATFORMS="gpu,cpu"

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
  'dm-tree==0.1.8' \
  'matplotlib==3.8.0' \
  'ml-collections==0.1.0' \
  'numpy==1.26.4' \
  'pandas==2.2.3' \
  'pytest<8.5.0' \
  'scipy==1.14.1' \
  'setuptools<72.0.0' \
  'tensorflow-cpu==2.20.0' \
  # Someone republished this on PyPI, so we don't have to go through conda, hence the odd name
  'pdbfixer-wheel==1.11.0' \
  'openmm==8.3.1' \
  # This isn't actually an alphafold dependency, but we use it for eval and it's
  # small, so I'm throwing it in here for now.
  'DockQ'

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

COPY --link . /app/alphafold
# I assume this tiny text file isn't just included upstream because of its GPL license
ADD --link \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt \
  /app/alphafold/alphafold/common/

WORKDIR /app/alphafold
