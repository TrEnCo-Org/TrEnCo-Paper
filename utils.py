import joblib as jl
import itertools
from copy import deepcopy

import pandas as pd
import numpy as np

class GridGenerator:
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def fit(self, data: pd.DataFrame):
        self.data = deepcopy(data)
        
        self.cols = data.columns
        
        # Get the binary columns
        self.bin_cols = []
        for col in self.cols:
            df = data[col]
            df = pd.to_numeric(df, errors='coerce')
            if df.notnull().all():
                if set(map(int, df.unique())) == {0, 1}:
                    data[col] = df.astype(int)
                    self.bin_cols.append(col)

        # Get the numerical columns
        self.num_cols = []
        for col in self.cols:
            if col not in self.bin_cols:
                df = data[col]
                if df.dtype != 'category':
                    df = pd.to_numeric(df, errors='coerce')
                    if df.notnull().all():
                        data[col] = df
                        self.num_cols.append(col)

        # Get the categorical columns
        self.cat_cols = []
        for col in self.cols:
            if col not in self.bin_cols and col not in self.num_cols:
                self.cat_cols.append(col)

        self.X = pd.get_dummies(data, columns=self.cat_cols)
        return self
    
    def generate(self, ngrid):
        grid_dict = {}
        for col in self.cols:
            if col in self.bin_cols:
                grid_dict[col] = self.rng.integers(2, size=ngrid)
            elif col in self.num_cols:
                grid_dict[col] = self.rng.uniform(self.data[col].min(), self.data[col].max(), size=ngrid)
            elif col in self.cat_cols:
                grid_dict[col] = self.rng.choice(self.data[col].unique(), size=ngrid)
        grid = pd.DataFrame(grid_dict)
        return grid

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import trenco.pruning as pr
from trenco.pruning import predict

def evaluate(
    rf,
    w,
    u,
    name,
    X_prop_train,
    y_prop_train,
    y_prop_train_pred,
    X_train, y_train,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred,
    X_grid,
    y_grid_pred
):
    y_prune_prop_train_pred = predict(rf.estimators_, X_prop_train, w*u)
    y_prune_train_pred = predict(rf.estimators_, X_train, w*u)
    y_prune_test_pred = predict(rf.estimators_, X_test, w*u)
    y_prune_grid_pred = predict(rf.estimators_, X_grid, w*u)
    
    return {
        f'n-{name}-trees': np.sum(u),
        f'{name}-prune-train-acc': np.mean(y_prune_train_pred == y_train),
        f'{name}-prune-test-acc': np.mean(y_prune_test_pred == y_test),
        f'{name}-prune-prop-train-acc': np.mean(y_prune_prop_train_pred == y_prop_train),
        f'{name}-prune-train-fidelity': np.mean(y_prune_train_pred == y_train_pred),
        f'{name}-prune-test-fidelity': np.mean(y_prune_test_pred == y_test_pred),
        f'{name}-prune-prop-train-fidelity': np.mean(y_prune_prop_train_pred == y_prop_train_pred),
        f'{name}-prune-grid-fidelity': np.mean(y_prune_grid_pred == y_grid_pred)
    }

def run_prop(
    rf,
    w,
    prop,
    X_train,
    y_train,
    y_train_pred,
    X_test,
    y_test,
    y_test_pred,
    X_grid,
    y_grid_pred,
    timelimit,
    verbose
):
    n_estimators = len(rf.estimators_)
    n = len(X_train)
    n_prop_train = int(prop * n)
    X_prop_train, y_prop_train = X_train[:n_prop_train], y_train[:n_prop_train]
    y_prop_train_pred = predict(rf.estimators_, X_prop_train, w)

    row = {}
    row['prune-prop'] = prop
    row['n-prune-prop-samples'] = n_prop_train
    
    # MIP:
    mip_pruner = pr.PrunerMIP(rf.estimators_)
    mip_pruner.set_gurobi_param('TimeLimit', timelimit)
    mip_pruner.set_gurobi_param('OutputFlag', verbose)
    
    u = mip_pruner.prune(X_prop_train)
    n_pruned_estimators = np.sum(u) # type: ignore
    
    row.update(evaluate(
        rf,
        w,
        u,
        'mip',
        X_prop_train,
        y_prop_train,
        y_prop_train_pred,
        X_train, y_train,
        y_train_pred,
        X_test,
        y_test,
        y_test_pred,
        X_grid,
        y_grid_pred
    ))
    row['mip-time'] = mip_pruner.st
    
    # Random:
    s = np.random.choice(n_estimators, n_pruned_estimators, replace=False)
    ur = np.zeros(n_estimators)
    ur[s] = 1
    
    row.update(evaluate(
        rf,
        w,
        ur,
        'random',
        X_prop_train,
        y_prop_train,
        y_prop_train_pred,
        X_train, y_train,
        y_train_pred,
        X_test,
        y_test,
        y_test_pred,
        X_grid,
        y_grid_pred
    ))
    
    # Greedy:
    greedy_pruner = pr.PrunerGreedy(rf.estimators_, n_pruned_estimators)
    ug = greedy_pruner.prune(X_prop_train)
    
    row.update(evaluate(
        rf,
        w,
        ug,
        'greedy',
        X_prop_train,
        y_prop_train,
        y_prop_train_pred,
        X_train, y_train,
        y_train_pred,
        X_test,
        y_test,
        y_test_pred,
        X_grid,
        y_grid_pred
    ))
    
    return row

def run_fold(
    X,
    y,
    fold,
    train_index,
    test_index,
    X_grid,
    n_estimators,
    n_props,
    timelimit,
    verbose
):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    
    rf.fit(X_train, y_train)
    w = np.ones(n_estimators)
    
    y_train_pred = predict(rf.estimators_, X_train, w)
    y_test_pred = predict(rf.estimators_, X_test, w)
    y_grid_pred = predict(rf.estimators_, X_grid, w)
    
    fold_row = {}
    fold_row['fold'] = fold+1
    fold_row['n-train-samples'] = len(X_train)
    fold_row['n-test-samples'] = len(X_test)
    fold_row['n-trees'] = len(rf.estimators_)
    fold_row['train-acc'] = np.mean(y_train_pred == y_train)
    fold_row['test-acc'] = np.mean(y_test_pred == y_test)
    
    rows = jl.Parallel(n_jobs=n_props)(
        jl.delayed(run_prop)(
            rf,
            w,
            prop,
            X_train,
            y_train,
            y_train_pred,
            X_test,
            y_test,
            y_test_pred,
            X_grid,
            y_grid_pred,
            timelimit,
            verbose
        )
        for prop in np.arange(1/n_props, 1 + 1/n_props, 1/n_props)
    )
    return list(map(lambda row: {**fold_row, **row}, rows)) # type: ignore
    
def run_dataset(
    name,
    data,
    target,
    n_splits: int = 4,
    n_props: int = 4,
    n_grid: int = 1000,
    n_estimators: int = 100,
    timelimit: int = 60,
    verbose: bool = True
):
    # Fit the GridGenerator to the dataset
    GG = GridGenerator(seed=42)
    GG.fit(data)
    X = GG.X
    X = X.values

    # Preprocess the dataset
    y = target.astype('category').cat.codes
    y = np.array(y.values)

    # Generate the grid
    data_grid = GG.generate(n_grid)

    # Transform the grid to the same format as the training set.
    X_grid = pd.get_dummies(data_grid, columns=GG.cat_cols)

    # Add the missing columns
    cols = list(set(GG.X.columns) - set(X_grid.columns))
    
    if len(cols) > 0:
        df = pd.DataFrame(0, index=np.arange(X_grid.shape[0]), columns=cols)
        X_grid = pd.concat([X_grid, df], axis=1)

    # Reorder the columns
    X_grid = X_grid[GG.X.columns]
    X_grid = X_grid.values

    dataset_row = {}
    dataset_row['dataset'] = str(name).lower()
    dataset_row['n-features'] = GG.data.shape[1]
    dataset_row['n-classes'] = len(target.unique())
    dataset_row['n-numerical-features'] = len(GG.num_cols)
    dataset_row['n-categorical-features'] = len(GG.cat_cols)
    dataset_row['n-binary-features'] = len(GG.bin_cols)
    dataset_row['n-grid-samples'] = n_grid

    # Split the dataset
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = jl.Parallel(n_jobs=n_splits)(
        jl.delayed(run_fold)(
            X,
            y,
            fold,
            train_index,
            test_index,
            X_grid,
            n_estimators,
            n_props,
            timelimit,
            verbose
        )
        for (
            fold,
            (train_index, test_index)
        ) in enumerate(kf.split(X))
    )
    
    rows = list(rows)
    rows = list(itertools.chain(*rows)) # type: ignore
    return list(map(lambda row: {**dataset_row, **row}, rows)) # type: ignore