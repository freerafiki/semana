import ollama
import numpy as np 
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def embed_list(a_list, ollama_emb_model, emb_size):
    X = np.zeros((len(a_list), emb_size))
    for i, string_data in enumerate(a_list):
        resp = ollama.embeddings(model=ollama_emb_model, prompt=string_data)
        X[i, :] = resp["embedding"]

    return X

# 3. Project any new document embedding onto the axis - this is the
#    whole "scoring" operation, a single dot product:
def project(x_doc, model):
    return np.dot(x_doc, model.coef_) + model.intercept_

def project_reduced(x_doc, pca, model):
    return np.dot(pca.transform(x_doc.reshape(1, -1)), model.coef_)[0] + model.intercept_

def project_poly(x_doc, pca, poly_f, model):
    x_pca = pca.transform(x_doc)
    x_poly = poly_f.transform(x_pca)
    score = m.predict(x_poly)
    return score 

def project_isotonic(x_doc, linear_model, iso):
    raw = linear_model.predict(x_doc.reshape(1, -1))
    return iso.predict(raw)[0]

# Scoring a new document, WITH uncertainty:
def project_gp(x_doc, pca_gp, gp):
    x_reduced = pca_gp.transform(x_doc.reshape(1, -1))
    mean, std = gp.predict(x_reduced, return_std=True)
    return mean[0], std[0]

# Nested LOO evaluation (redo PCA+poly+ridge per fold to avoid leakage)
def loo_poly_eval(X, y, n_components):
    preds = np.zeros(len(y))
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        pca_f = PCA(n_components=PCA_COMPONENTS).fit(X[train_idx])
        Xtr, Xte = pca_f.transform(X[train_idx]), pca_f.transform(X[test_idx])
        poly_f = PolynomialFeatures(degree=2, include_bias=False).fit(Xtr)
        Xtr_p, Xte_p = poly_f.transform(Xtr), poly_f.transform(Xte)
        m = RidgeCV(alphas=alphas).fit(Xtr_p, y[train_idx])
        preds[test_idx] = m.predict(Xte_p)
    return preds

# Nested LOO evaluation (refit both stages per fold)
def loo_isotonic_eval(X, y):
    preds = np.zeros(len(y))
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        lin = RidgeCV(alphas=alphas).fit(X[train_idx], y[train_idx])
        raw_tr = lin.predict(X[train_idx])
        raw_te = lin.predict(X[test_idx])
        iso_f = IsotonicRegression(out_of_bounds='clip').fit(raw_tr, y[train_idx])
        preds[test_idx] = iso_f.predict(raw_te)
    return preds

# Nested LOO evaluation
def loo_gp_eval(X, y, n_components):
    preds = np.zeros(len(y))
    stds = np.zeros(len(y))
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        pca_f = PCA(n_components=n_components).fit(X[train_idx])
        Xtr, Xte = pca_f.transform(X[train_idx]), pca_f.transform(X[test_idx])
        k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-2, (1e-5, 1e1))
        m = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=10)
        m.fit(Xtr, y[train_idx])
        pred, std = m.predict(Xte, return_std=True)
        preds[test_idx] = pred
        stds[test_idx] = std
    return preds, stds

def evaluate(embeddings, labels, model, projection_method, print_results=True, \
             pca=None, reduced=False, poly_f=None, iso=None):
    """
    projection_func is callable (the projection function)
    """
    scores = np.zeros((len(embeddings), 1))
    errors = np.zeros((len(embeddings), 1))
    min_val = 1
    max_val = 0

    assert len(embeddings) == len(labels), "misaligned embeddings and labels!"
    
    for i, test_sentence in enumerate(embeddings):
        if projection_method == 'ridge':
            if reduced:
                score = project_reduced(test_sentence, pca, model)
            else:
                score = project(test_sentence, model)
        elif projection_method == 'polynomial':
            score = project_poly(test_sentence, pca=pca, poly_f=poly_f, model=model)
        elif projection_method == 'isotonic':
            score = project_isotonic(test_sentence, linear_model=model, iso=iso)
        elif projection_method == 'gaussian':
            score = project_gp(test_sentence, pca_gp=pca, gp=model)
        scores[i] = score
        if score < min_val:
            min_val = score 
        if score > max_val:
            max_val = score
        errors[i] = np.abs(score - labels[i])

    if print_results:
        print(f"    MAE: {(np.mean(errors)):.03f}")
        print(f"    MSE: {(np.mean(np.square(errors))):.03f}")
        print(f"    RMSE: {(np.mean(np.sqrt(np.square(errors)))):.03f}")
        print(f"    min_val: {min_val:.03f}")
        print(f"    max,_val: {max_val:.03f}")

    return scores, errors, min_val, max_val


def evaluate_probe(y_true, y_pred, set_name=""):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. R² — proportion of variance explained (higher is better, max=1)
    r2 = r2_score(y_true, y_pred)
    
    # 2. Mean Absolute Error — average distance from true label (lower is better)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 3. Spearman rank correlation — monotonic ordering (higher is better, max=1)
    #    Checks if predictions are in the right ORDER, even if values are compressed
    spearman, _ = spearmanr(y_true, y_pred)
    
    # 4. Max absolute error — worst-case prediction (lower is better)
    max_err = np.max(np.abs(y_true - y_pred))
    
    print(f"\n{'─' * 40}")
    print(f"  {set_name} Metrics")
    print(f"{'─' * 40}")
    print(f"  R² Score:            {r2:.4f}   (>0.7 good, >0.5 acceptable)")
    print(f"  Mean Absolute Error: {mae:.4f}   (<0.15 good, <0.25 acceptable)")
    print(f"  Spearman ρ:         {spearman:.4f}   (closer to 1 = correct ordering)")
    print(f"  Max Absolute Error:  {max_err:.4f}   (worst single prediction)")
    print(f"{'─' * 40}")
    
    return { "r2": r2, "mae": mae, "spearman": spearman, "max_err": max_err}

