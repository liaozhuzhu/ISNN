# ISNN

> **Official implementation of the paper:**  
> **"Implicit Subgraph Neural Network"**  
> Yongjian Zhong, Liao Zhu, Hieu Vu, Bijaya Adhikari  
> _Accepted at the 42nd International Conference on Machine Learning (ICML 2025), Poster_
> [[OpenReview PDF](https://openreview.net/pdf?id=QhCb3FAQi2)]

## Setup

### 1. Install Required Libraries

To begin using the ISNN framework, ensure all necessary libraries are installed. You can install the required libraries by using the `requirements.txt` file. To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```
or use your preferred package manager.

This repository was run and tested with Python 3.8.10.

### 2. Datasets

The **Real-World Datasets** used in this project includes `HPO_METAB`, `HPO_NEURO`, `PPI_BP`, and `EM_USER`, all of which were sourced from the paper:

**"Subgraph Neural Networks"**
Emily Alsentzer, Samuel G. Finlayson, Michelle M. Li, Marinka Zitnik  
_Proceedings of Neural Information Processing Systems (NeurIPS), 2020_ [[arXiv](https://arxiv.org/abs/2006.10538)]

and can be downloaded from this Dropbox link [here](https://www.dropbox.com/scl/fo/hbjyz991xifmuccfk82hh/AEq9fYl_Ed4TOWlnifJsJ5w?rlkey=zpyvstdbjbwyfmi3e2l2qtqih&e=1&dl=0).

Once installed, unzip the contents of the folder and set the `DATASET_PATH` variable in the `config_path.py` file to the path of the folder containing the datasets.

### 3. Plotting Results

Inside the root (`ISNN`) directory, create a directory called `plots` with a subdirectory for each dataset you are working with. For example, to create the directories for the `HPO_METAB` dataset, run the following commands:

```bash
cd ~/PATH/TO/ISNN
mkdir plots
cd plots
mkdir hpo_metab
```

Once this is setup, you can proceed to [Running](#running) the ISNN framework.

## Running

To run ISNN, make sure you are cd'd into the `ISNN` directory and run the following command:

```bash
python ISNN.py --dataset {dataset_name} --model isnn --repeat 10
```

The hyperparameters used to produce the results in the paper are preset and can be found in the `/hyperparams/{dataset}.yml` file. To change the hyperparameters, you can modify the `.yml` file.

## Results

A summary of the results will be written to `{dataset_name}_{model}_results.json` in the root directory of the project.
