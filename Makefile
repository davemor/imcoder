#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = imcoder
PWD := $(shell pwd)

# docker options
DOCKER_IMAGE_NAME = imcoder

# docker build process info
LOCAL_USER = $(USER)
LOCAL_UID = $(shell id -u)
LOCAL_GID = $(shell id -g)

#################################################################################
# LOCAL ENV COMMANDS                                                            #
#################################################################################
environment:
	python -m venv venv
	. venv/bin/activate &&  pip install -r requirements.txt && pip install -e .
	echo "Remember to activate the environment before use: . venv/bin/activate"


#################################################################################
# CONTAINER COMMANDS                                                            #
#################################################################################
image:
	docker build -t $(DOCKER_IMAGE_NAME) .

run_eln:
	docker run --shm-size=64G \
				--gpus all \
				--rm \
				--name $(USER)-$(PROJECT_NAME) \
				--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
				--network=host \
				-v /home/$(LOCAL_USER)/repos/$(PROJECT_NAME):/workspace/repos/$(PROJECT_NAME) \
				-v /data2:/data \
				-v /nvme:/scratch \
				-it $(DOCKER_IMAGE_NAME)
