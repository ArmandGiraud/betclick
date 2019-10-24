# Betclick Churn Analysis
This repo contains an Assignment for betclick

### Install:
 - clone the repo
 - `cd ./src`

### Fit

- fit the model on data using:
    ```python main.py```

The script will:
    - download the data, and ask a password for unzipping
    - label the data and drop leaky rows
    - preprocess and write serializables necessary for inference on disk

### Predict

`python main.py --predict`

- predict whether each customer in a subsample of the dataset is a potential churner.
- write a file on disk in the preds folder
the first column rerpresents customer_key
the second column is the target 

| customer_key | is_churner |
|--------------|------------|
| 10390929     | True       |
| 10390926     | True       |
| 10390926     | False      |
| 10390926     | True       |
| 10390926     | False      |
| 10390926     | False      |

#### Predict on private dataset

`python main.py --predict --private_file "my_data_file.csv"`

The data should have the same format as the original one, it might be necessary to handle mix typed columns.