from pathlib import Path

import joblib as jl
import trenco as tr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

current_dir = Path(__file__).parent
datasets_dir = current_dir / 'datasets'

# Get all directories in the datasets directory
datasets = [
    d for d in datasets_dir.iterdir()
    if d.is_dir()]

# For each dataset
ne = 100

def run_prop(
    rf,
    X_train,
    **kwargs
):
    return tr.pruning.prune_mip_exact(
        rf, X_train, **kwargs)


def run(
    dataset,
    ne: int = 100,
    folder = current_dir / 'results' / 'on-train-acc-vs-prop'
):
    # Load the dataset
    full_path = dataset / f'{dataset.name}.full.csv'
    full = pd.read_csv(full_path)
    
    # Get the class column
    klass = full.columns[-1]
    
    # Separate the X and y
    # drop the last column to get the features
    X_full = full.drop(klass, axis=1).values
    y_full = full[klass].values
    y_full = np.array(y_full)
    
    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42)
    
    # Train a random forest model on the data
    rf = RandomForestClassifier(
        n_estimators=ne, random_state=42)
    
    # Fit the model
    rf.fit(X_train, y_train)
    
    props = np.arange(0.1, 1.1, 0.1)
    n = len(X_train)
    
    def run_prop_on_p(p):
        return p, run_prop(
            rf, X_train[:int(n*p)],
            timelimit=600,
            verbose=True)
    
    us = jl.Parallel(n_jobs=-1)(
        jl.delayed(run_prop_on_p)(p) for p in props)
    us = list(us)

    fn = dataset.name.lower()
    y_pred = tr.ensemble.predict(rf, X_train, np.ones(len(rf)))
    test_acc = np.mean(y_pred == y_train)

    rows = []
    for p, u in us: # type: ignore
        y_pred = tr.ensemble.predict(rf, X_test, u)
        prune_test_acc = np.mean(y_pred == y_test)
        nt = sum(u)
        row = {
            'prop': p,
            'test-acc': test_acc,
            'prune-test-acc': prune_test_acc,
            'n-trees': nt
        }
        
        rows.append(row)
      
    df = pd.DataFrame(rows)
    df.to_csv(folder / f'{fn}.csv', index=False)

def main():
    jl.Parallel(n_jobs=-1)(
        jl.delayed(run)(dataset, ne) for dataset in datasets)

if __name__ == '__main__':
    main()