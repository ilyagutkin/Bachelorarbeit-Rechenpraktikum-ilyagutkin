# taken from Pedro Costa Klein (p.klein@math.uni-goettingen.de)
# see https://gitlab.gwdg.de/crc1456/livedocs/livedocs_template
FROM ubuntu:latest
LABEL maintainer="Christoph Lehrenfeld" 


# Initial ubuntu update
RUN apt update

# Install required packages
RUN apt install -y \
	python3 \
	python3-pip

# Install dependencies
RUN python3 -m pip install --no-cache-dir notebook jupyterlab
RUN pip install --no-cache-dir jupyterhub
RUN pip install voila

# Expose the port used by the jupyter notebook server
EXPOSE 8888

# create user with a home directory
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}

# Copy your repo directory
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# Install the requirements from requirements.txt
RUN pip install -r requirements.txt

# Start the jupyter notebook inside the repo folder
CMD ["jupyter" , "notebook" , "--ip='0.0.0.0'" , "--port=8888", "--NotebookApp.token='methodsnm'"]
	
