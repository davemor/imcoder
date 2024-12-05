#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = imcoder
PWD := $(shell pwd)

# docker options
DOCKER_IMAGE_NAME = imcoder

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

run:
	docker run --shm-size=64G \
				--gpus all \
				--rm \
				--name $(USER)-$(PROJECT_NAME) \
				--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
				--network=host \
				-v $(PWD):/workspace/repos/$(PROJECT_NAME) \
				-it $(DOCKER_IMAGE_NAME)
