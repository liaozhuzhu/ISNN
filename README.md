# ISNN

## Setup

### 1. Install Required Libraries

To begin using the ISNN framework, ensure all necessary libraries are installed. You can install the required libraries by using the `requirements.txt` file. To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```
or use your preferred package manager.

### 2. Datasets

The **Real-World Datasets** used in this project were sourced from the paper:

```txt
**"Subgraph Neural Networks"**  
Emily Alsentzer, Samuel G. Finlayson, Michelle M. Li, Marinka Zitnik  
_Proceedings of Neural Information Processing Systems (NeurIPS), 2020_  
```

and can be downloaded from this Dropbox link [here](https://www.dropbox.com/scl/fo/hbjyz991xifmuccfk82hh/AEq9fYl_Ed4TOWlnifJsJ5w?rlkey=zpyvstdbjbwyfmi3e2l2qtqih&e=1&dl=0).

Once installed, unzip the contents of the folder and set the `DATASET_PATH` variable in the `config_path.py` file to the path of the folder containing the datasets.

## Running

To run ISNN, make sure you are cd'd into the `ISNN` directory and run the following command:

```bash
python ISNN.py --dataset hpo_metab
```

The hyperparameters used to produce the results in the paper are preset and can be found in the `/hyperparams/{dataset}.yml` file. To change the hyperparameters, you can modify the `.yml` file or pass the hyperparameters as arguments to the command line.