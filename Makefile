# Image names
API_IMAGE=notag.azurecr.io/notag-ml-api:latest
TRAIN_IMAGE=notag.azurecr.io/notag-ml-trainer:latest
SERVE_IMAGE=notag.azurecr.io/notag-ml-serve:latest

.PHONY: all api training

all: api training

api:
	docker build -f api/Dockerfile -t $(API_IMAGE) .

training:
	docker build -f training/Dockerfile -t $(TRAIN_IMAGE) .

serve:
	docker build -f serve/Dockerfile -t $(SERVE_IMAGE) .

push-azure:
	az login
	az acr login --name notag
	docker push $(API_IMAGE)
	docker push $(TRAIN_IMAGE)