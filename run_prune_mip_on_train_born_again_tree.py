from pathlib import Path
from itertools import chain

import time
import joblib as jl

import trenco as tr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

current_dir = Path(__file__).parent
datasets_dir = current_dir / 'datasets'

# Get all directories in the datasets directory
datasets = [
    d for d in datasets_dir.iterdir()
    if d.is_dir()]

def run_on_i(
    dataset,
    i: int,
    ne: int = 100,
    timelimit: int = 600,
    nr: int = 5
):
    train_path = dataset / f'{dataset.name}.train{i}.csv'
    test_path = dataset / f'{dataset.name}.test{i}.csv'
    
    # Load the training data
    train = pd.read_csv(train_path)
    
    # Get the class column
    klass = train.columns[-1]
    
    # Separate the X and y
    # drop the last column to get the features
    X_train = train.drop(klass, axis=1).values
    y_train = train[klass].values
    y_train = np.array(y_train)
    
    # Load the test data
    test = pd.read_csv(test_path)
    X_test = test.drop(klass, axis=1).values
    y_test = test[klass].values
    
    # Train a random forest model on the data
    rf = RandomForestClassifier(
        n_estimators=ne, random_state=42)
    
    # Fit the model
    rf.fit(X_train, y_train)
    
    # Evaluate the initial model on the training data
    u = np.ones(len(rf))
    y_pred = tr.ensemble.predict(rf, X_train, u)
    train_acc = np.mean(y_pred == y_train)
    
    # Evaluate the initial model on the test data
    y_pred = tr.ensemble.predict(rf, X_test, u)
    test_acc = np.mean(y_pred == y_test)
    
    # Prune the model using MIP
    # With the time limit for the
    # MIP solver set to timelimit
    start = time.time()
    u = tr.pruning.prune_mip_exact(
        rf, X_train,
        verbose=True,
        timelimit=timelimit)
    end = time.time()
    t = end - start
    
    # Compute the accuracy of the pruned model on the training data
    y_pred = tr.ensemble.predict(rf, X_train, u)
    prune_train_acc = np.mean(y_pred == y_train)
    
    # Evaluate the pruned model on the test data
    y_pred = tr.ensemble.predict(rf, X_test, u)
    prune_test_acc = np.mean(y_pred == y_test)
    
    # Evaluate the fidelity of the pruned model
    # on the training data
    y_pruned_pred = tr.ensemble.predict(rf, X_train, u)
    y_pred = tr.ensemble.predict(rf, X_train, np.ones(len(rf)))
    prune_train_fidelity = np.mean(y_pred == y_pruned_pred)
    
    # Evaluate the fidelity of the pruned model
    # on the test data
    y_pruned_pred = tr.ensemble.predict(rf, X_test, u)
    y_pred = tr.ensemble.predict(rf, X_test, np.ones(len(rf)))
    prune_test_fidelity = np.mean(y_pred == y_pruned_pred)
    
    # Number of trees in the pruned model
    nt = sum(np.array(u))
    
    # Random Pruning
    random_prune_train_acc = 0
    random_prune_test_acc = 0
    random_prune_train_fidelity = 0
    random_prune_test_fidelity = 0
    
    for _ in range(nr):
        sr = np.random.choice(len(rf), nt, replace=False)
        ur = np.zeros(len(rf))
        ur[sr] = 1
        
        # Evaluate the random pruned model on the training data
        y_pred = tr.ensemble.predict(rf, X_train, ur)
        random_prune_train_acc += np.mean(y_pred == y_train)
        
        # Evaluate the random pruned model on the test data
        y_pred = tr.ensemble.predict(rf, X_test, ur)
        random_prune_test_acc += np.mean(y_pred == y_test)
        
        # Evaluate the fidelity of the random pruned model
        # on the training data
        y_pruned_pred = tr.ensemble.predict(rf, X_train, ur)
        y_pred = tr.ensemble.predict(rf, X_train, np.ones(len(rf)))
        random_prune_train_fidelity += np.mean(y_pred == y_pruned_pred)
        
        # Evaluate the fidelity of the random pruned model
        # on the test data
        y_pruned_pred = tr.ensemble.predict(rf, X_test, ur)
        y_pred = tr.ensemble.predict(rf, X_test, np.ones(len(rf)))
        random_prune_test_fidelity += np.mean(y_pred == y_pruned_pred)

    random_prune_train_acc /= nr
    random_prune_test_acc /= nr
    random_prune_train_fidelity /= nr
    random_prune_test_fidelity /= nr
    
    # Greedy Pruning
    ug = tr.pruning.greedy.prune_greedy_exact(
        rf, X_test, y_test, k=nt)
    
    # Evaluate the greedy pruned model on the training data
    y_pred = tr.ensemble.predict(rf, X_train, ug)
    greedy_prune_train_acc = np.mean(y_pred == y_train)
    
    # Evaluate the greedy pruned model on the test data
    y_pred = tr.ensemble.predict(rf, X_test, ug)
    greedy_prune_test_acc = np.mean(y_pred == y_test)
    
    # Evaluate the fidelity of the greedy pruned model
    # on the training data
    y_pruned_pred = tr.ensemble.predict(rf, X_train, ug)
    y_pred = tr.ensemble.predict(rf, X_train, np.ones(len(rf)))
    greedy_prune_train_fidelity = np.mean(y_pred == y_pruned_pred)
    
    # Evaluate the fidelity of the greedy pruned model
    # on the test data
    y_pruned_pred = tr.ensemble.predict(rf, X_test, ug)
    y_pred = tr.ensemble.predict(rf, X_test, np.ones(len(rf)))
    greedy_prune_test_fidelity = np.mean(y_pred == y_pruned_pred)
        
    row = {
        'dataset': dataset.name,
        'index': i,
        'n-train-samples': len(X_train),
        'n-test-samples': len(X_test),
        'n-trees': nt,
        'train-acc': train_acc,
        'test-acc': test_acc,
        'prune-train-acc': prune_train_acc,
        'prune-test-acc': prune_test_acc,
        'prune-train-fidelity': prune_train_fidelity,
        'prune-test-fidelity': prune_test_fidelity,
        'random-prune-train-acc': random_prune_train_acc,
        'random-prune-test-acc': random_prune_test_acc,
        'random-prune-train-fidelity': random_prune_train_acc,
        'random-prune-test-fidelity': random_prune_test_fidelity,
        'greedy-prune-train-acc': greedy_prune_train_acc,
        'greedy-prune-test-acc': greedy_prune_test_acc,
        'greedy-prune-train-fidelity': greedy_prune_train_fidelity,
        'greedy-prune-test-fidelity': greedy_prune_test_fidelity,
        'time': t
    }

    return row

def run_on(
    dataset,
    ne: int = 100,
    nd: int = 5,
    timelimit: int = 600,
    parallel: bool = True
):
    if not parallel:
        rows = []
        for i in range(1, nd+1):
            row = run_on_i(
                dataset, i,
                ne, timelimit)
            rows.append(row)
    else:
        rows = jl.Parallel(n_jobs=nd)(
            jl.delayed(run_on_i)(
                dataset, i,
                ne, timelimit)
            for i in range(1, nd+1))
        rows = list(rows)
    return rows

nd = 5
ne = 100
timelimit = 600

def main():
    rows = jl.Parallel(n_jobs=len(datasets))(
        jl.delayed(run_on)(
            dataset, ne, nd, timelimit)
        for dataset in datasets)
    rows = list(rows)
    df = pd.DataFrame(list(chain(*rows))) # type: ignore

    df.to_csv(current_dir / 'results' / 'prune-mip-on-train-born-again-tree.csv', index=False)

if __name__ == '__main__':
    main()