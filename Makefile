# build genmodels container
build:
	docker build -t farazrahman121/ml-monorepo .

# pull from Dockerhub
pull-image:
	docker pull farazrahman121/ml-monorepo

# push to Dockerhub
push-image:
	docker pull farazrahman121/ml-monorepo

# Changes will not be reflected
run:
	docker run --gpus all -it farazrahman121/ml-monorepo

# Changes will reflect in container and persist in repo
run-dev:
	docker run \
		--gpus all \
		--shm-size=32g \
		-it \
		-v $(shell pwd):/workspace farazrahman121/ml-monorepo

# Build simple Python PyTorch CPU container
build-simple:
	docker build -f Dockerfile.simple -t python-pytorch-cpu .

# Run simple Python PyTorch CPU container with shell
run-simple:
	docker run -it -v $(shell pwd):/app python-pytorch-cpu