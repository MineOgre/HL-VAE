Heterogeneous Longitudinal Variational Autoencoder
===========================================================================================
This repository contains the python scripts used in our [paper]().

Downloading MNIST digits
------------------------
- Download and unzip archive from here: https://www.dropbox.com/s/j80vfwcqqu3vmnf/trainingSet.tar?dl=0
- Do not push to remote repository.

Generating Health MNIST experiment data
---------------------------------------
- To create training/test data, labels as well as mask, run:
		`python Heterogeneous_Health_MNIST_generate.py --source=./trainingSet --destination=./data --num_3=10 --num_6=10 --missing=25 --data_file_name=train.csv --labels_file_name=labels.csv --mask_file_name=mask.csv --data_masked_file_name=masked_data.csv ----datatype_config=D3`
- See `Heterogeneous_Health_MNIST_generate.py` for configuration.

Using our method
----------------
-To run the **HLVAE model**:
        `python HLVAE_main.py --f=./config/hlvae_config_file.txt`
