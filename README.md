A webapp for submitting kaggle urls and building neural networks.
Currently limited by $150 monthly visual studio enterprise credit and lack of gpu enabled container availability for subscription. Training is slow and favors longer compute time over large batches to save on maximum memory of 16gb

General flow:

/
api returns hello world

/train:
api receives kaggle url -> api spins up an azure container instance (ACI) to train the model -> Once the ACI trains a model, saves that model to a storage blob and shuts off

/models:
api queries blob for available models and their source kaggle url

/serve:
api receives a kaggle url that is in storage blob with a trained model -> ACI spins up container to host a model for inference -> After 10 minutes, shutdown the container
