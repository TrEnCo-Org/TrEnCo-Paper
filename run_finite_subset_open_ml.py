import joblib as jl
from pathlib import Path

# Run it sequentially
n_splits = 5
n_props = 10
n_grid = 10000
n_estimators = 100
timelimit = 600
verbose = False

# For each dataset:
# 1. Load the dataset
# 2. Preprocess the dataset:
#    a. Standardize the features
#    b. Create a random generator for the dataset
# 2. Randomly generate `ngrid` samples
# 3. split the dataset to `nfolds` folds
# 4. For each fold:
#    a. Take 4 folds as training set and 1 fold as test set
#    b. Train a model on the training set with `nestimators` estimators
#    c. Evaluate the model on the test set
#    d. Split the training set to `nprops` proportions
#    e. For each proportion:
#       i. Take the proportion of the training set
#      ii. Prune the model on the proportion
#     iii. Evaluate the pruned model on the propotion train set
#      iv. Evaluate the pruned model on the full train set
#       v. Evaluate the pruned model on the test set

from sklearn.datasets import fetch_openml

from utils import *

def run(dataset_id, folder: Path):
    # Load the dataset
    dataset = fetch_openml(data_id=dataset_id, as_frame=True)
    name = dataset.details['name']
    name = name.replace(' ', '-')
    name = name.replace('_', '-')
    data, target = dataset.data, dataset.target

    rows = list(run_dataset(
        name,
        data,
        target,
        n_splits=n_splits,
        n_props=n_props,
        n_grid=n_grid,
        n_estimators=n_estimators,
        timelimit=timelimit,
        verbose=True
    ))
    
    results_file = folder / f'{dataset_id}.csv'
    results = pd.DataFrame(rows)
    results.to_csv(results_file, index=False)

datasets = [
    44, # spambase
    # 4135, # Amazon_employee_access
    40982, # steel-plates-fault
    41703, # MIP-2016-classification
    43098, # Students_scores
    43672, # Heart-Disease-Dataset-(Comprehensive)
    45036, # default-of-credit-card-clients
    45058, # credit-g
    45068, # Adult
    45578, # California-Housing-Classification
]

results_dir = Path('__file__').parent / 'results'
folder = results_dir / ('finite_subset_open_ml' + str(2))
folder.mkdir(exist_ok=True)

jl.Parallel(n_jobs=-1)(
    jl.delayed(run)(
        dataset_id,
        folder
    )
    for dataset_id in datasets
)
