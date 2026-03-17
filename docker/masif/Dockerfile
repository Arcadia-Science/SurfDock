FROM python:3.9-bullseye
WORKDIR /root/
ARG NUM_CORES=14

# ── 1. System deps for PyMesh build ─────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc g++ git cmake \
    libgmp-dev libmpfr-dev libgmpxx4ldbl \
    libboost-dev libboost-thread-dev \
    zip unzip patchelf && \
    apt-get clean

# ── 2a. Clone PyMesh + submodules ───────────────────────────────────────────
RUN git clone --single-branch -b main https://github.com/PyMesh/PyMesh.git
ENV PYMESH_PATH=/root/PyMesh
ENV NUM_CORES=$NUM_CORES
WORKDIR $PYMESH_PATH
RUN git submodule update --init

# ── 2b. Build PyMesh wheel (slow, ~15 min) ──────────────────────────────────
ENV CFLAGS="-fcommon"
ENV CXXFLAGS="-fcommon"
RUN pip install -r $PYMESH_PATH/python/requirements.txt && \
    ./setup.py bdist_wheel && \
    rm -rf build_3.9 third_party/build && \
    python $PYMESH_PATH/docker/patches/patch_wheel.py dist/pymesh2*.whl && \
    pip install dist/pymesh2*.whl

# ── 2c. Pin numpy for pymesh compatibility (Tester removed in 1.25) ─────────
RUN pip install "numpy<1.25" && \
    python -c "import pymesh; print('pymesh', pymesh.__version__)"

# ── 3. APBS 3.4.1 + pdb2pqr 2.1.1 + MSMS (from SurfDock tarball) ──────────
COPY APBS_PDB2PQR.tar.gz /tmp/
RUN cd /tmp && tar xzf APBS_PDB2PQR.tar.gz && \
    mv /tmp/transfer/APBS-3.4.1.Linux /opt/APBS && \
    mv /tmp/transfer/pdb2pqr-linux-bin64-2.1.1 /opt/pdb2pqr && \
    rm -rf /tmp/transfer /tmp/APBS_PDB2PQR.tar.gz

ENV APBS_BIN=/opt/APBS/bin/apbs
ENV MULTIVALUE_BIN=/opt/APBS/share/apbs/tools/bin/multivalue
ENV PDB2PQR_BIN=/opt/pdb2pqr/pdb2pqr
ENV MSMS_BIN=/opt/APBS/bin/msms

# ── 4. reduce (protonation) ─────────────────────────────────────────────────
WORKDIR /tmp
RUN git clone https://github.com/rlabduke/reduce.git && \
    cd reduce && \
    make && \
    make install && \
    cd /tmp && rm -rf reduce

# ── 5. OpenBabel (CLI + dev headers + SWIG for Python bindings) ──────────────
RUN apt-get update && apt-get install -y libopenbabel-dev openbabel swig && apt-get clean

# ── 6. Python dependencies ──────────────────────────────────────────────────
RUN pip install "numpy<1.25" scipy scikit-learn biopython ipython && \
    pip install openbabel --global-option=build_ext \
        --global-option="-I/usr/include/openbabel3" \
        --global-option="-L/usr/lib/x86_64-linux-gnu"

# ── 7. MaSIF source ─────────────────────────────────────────────────────────
RUN git clone --single-branch https://github.com/LPDI-EPFL/masif.git /masif

WORKDIR /masif
CMD ["bash"]
