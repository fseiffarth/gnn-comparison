# Contributions to the original repository

This repository is a fork of [diningphil/gnn-comparison](https://github.com/diningphil/gnn-comparison).
The code contains the following contributions:

- Additional real-world datasets: DHFR, NCI109, Mutagenicity
- Additional featured datasets: DHFRFeatures, IMDB-BINARY-Features, IMDB-MULTI-Features, NCI109Features, MutagenicityFeatures
- Additional synthetic datasets: CSL, LongRings100 (RingTransfer1), EvenOddRingsCount (RingTransfer2), EvenOddRings2_16 (RingTransfer3), Snowflakes

## Reproduce the Results
To reproduce the part of experiments in the paper based on the evaluation proposed in the original repository follow the instructions below.
1. Clone this repository
2. Install the environment by running the following commands:
```bash
   conda env create -f environment.yml
   conda activate gnn-comparison
```
3. Unzip the `DATA.zip` folder and run the following commands for the different models

### Run experiments
Given dataset DATA and algorith ALGO run the following command:
```bash
export OMP_NUM_THREADS=1
python Launch_Experiments.py --config-file config_ALGO.yml --dataset-name DATA --result-folder RESULTS --inner-processes 1 --outer-processes 10
```



## Adding Custom Datasets

This is an instruction on how to run the experiments with new datasets.
New datasets need to be in the file format described below in steps 2 and 3.

1. Create a folder in the `DATA` directory with the NAME of the dataset and add `raw` and `processed` subfolders.
2. In the `raw` folder, add the following files:
    - `NAME_Edges.txt` containing the edges of the graph in the format `graph_id, source, target, edge_label`
    - `NAME_Nodes.txt` containing the nodes of the graph in the format `graph_id, node_id, node_label_1, node_label_2, ...`
    - `NAME_Labels.txt` containing the graph labels in the format `graph_dataset_name graph_id, graph_label`
3. In the `processed` folder, add the file `NAME_splits.json` containing the splits of the dataset in the format:
    ```json
   [{
           "test": [ 5, 6],
           "model_selection": [{"train": [1, 2, 4, 7, 9, 10], "val": [3, 8]}]
    
   },...]
    ```
4. Add the dataset as class in the [datasets/manager.py](datasets/manager.py) file using the following scheme replacing `NAME` with the name of the dataset:
```python
         class NAME(BenchmarkDatasetManager):
          name = "NAME"
          _dim_features = 1
          _dim_target = 10
          max_num_nodes = 41
```
_dim_features is the number of features per node, _dim_target is the number of classes, and max_num_nodes is the maximum number of nodes in the dataset.
5. Add the dataset to the dictionary 
   ```python
   datasets = {
    ...
    "NAME": NAME,
   }
   ```
   in [config/base.py](config/base.py) 
6. Add NAME to the list of datasets in [datasets/__init__.py](datasets/__init__.py)
7. In case you want to use the dataset for DGCNN, you need to manipulate the dictionary
   ```python
   self.ks = {
    ...
    'NAME': {'0.6': 135, '0.9': 180},
    }
   ```
   in [models/graph_classifiers/DGCNN.py](models/graph_classifiers/DGCNN.py).
   The number corresponding to '0.6' is the 0.6 median of the size of the graphs in the dataset and the number corresponding to '0.9' is the 0.9 median of the size of the graphs in the dataset.
8. Add the dataset to the dictionary
   ```python
   DATASETS = {
    ...
    'NAME': NAME,
    }
   ```
   in [PrepareDatasets.py](PrepareDatasets.py).
9. Preprocess the dataset NAME by running the following command:
   ```bash
   python PrepareDataset.py DATA --dataset-name NAME --outer-k 10
   ```
10. Run the experiments with the new dataset using the following commands:
    ```bash
    export OMP_NUM_THREADS=1
    python Launch_Experiments.py --config-file <config> --dataset-name NAME --result-folder RESULTS --inner-processes 3 --outer-processes 10
    ```
