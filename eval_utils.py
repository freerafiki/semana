import ollama
import numpy as np 
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr


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

def evaluate(embeddings, labels, model, projection_method, reduced=False, pca=None, print_results=True):
    """
    projection_func is callable (the projection function)
    """
    scores = np.zeros((len(embeddings), 1))
    errors = np.zeros((len(embeddings), 1))
    min_val = 1
    max_val = 0

    assert len(embeddings) == len(labels), "misaligned embeddings and labels!"
    
    for i, test_sentence in enumerate(embeddings):
        if reduced:
            score = project_reduced(test_sentence, pca, model)
        else:
            score = project(test_sentence, model)
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
        print(f"    ma},_val: {max_val:.03f}")

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

