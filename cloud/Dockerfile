# Container for Google Cloud Batch tasks (and for running the Python analysis).
# Build & push (Artifact Registry):
#   gcloud artifacts repositories create pellet --repository-format=docker --location=us-central1
#   docker build -f cloud/Dockerfile -t us-central1-docker.pkg.dev/PROJECT/pellet/pellet-pd:2 .
#   docker push us-central1-docker.pkg.dev/PROJECT/pellet/pellet-pd:2
# The solver is compiled at task start with -march=native (cloud/task.sh), so one
# image serves every machine family (C3D, C4D, N2D, ...).
FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:stable

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ libgomp1 python3 python3-pip python3-numpy python3-scipy python3-pandas \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pellet
COPY pellet_pd.cpp build.sh *.py ./
COPY cloud/task.sh cloud/task.sh
RUN chmod +x cloud/task.sh build.sh \
 && g++ -O3 -fopenmp -std=c++17 -DPD_BUILD="\"$(sha256sum pellet_pd.cpp | cut -c1-16)\"" pellet_pd.cpp -o /usr/local/bin/pellet_pd

ENTRYPOINT ["/bin/bash"]
