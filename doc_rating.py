import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import matplotlib.pyplot as plt

from anchor_sentences import REGULATED_MARKET_ANCHORS_SENTENCES as ANCHORS
from test_sentences import REGULATED_MARKET_TEST_SENTENCES as TEST_SET
from parameters import EMBEDDING_MODEL, EMBEDDING_SIZE, PCA_COMPONENTS, EVALUATE_PROBE
from eval_utils import embed_list, evaluate, evaluate_probe

##############################
# INPUT
##############################
# X: shape (N, 768) - embeddings of your N graded sentences
anchors = ANCHORS['data']
X = np.zeros((len(anchors), EMBEDDING_SIZE))

##############################
# EMBEDDINGS
##############################
print("=" * 50)
print(f"Embedding the ANCHOR sentences with {EMBEDDING_MODEL},.")
anchor_embeddings = embed_list(anchors, EMBEDDING_MODEL, EMBEDDING_SIZE)
# y: shape (N,)      - their Likert labels, e.g. [1, 1, 2, 3, 4, 5, 5, ...]
anchor_labels = ANCHORS['labels']
anchor_labels /= np.max(anchor_labels) # normalization

# if EVALUATE_PROBE:
#     print("=" * 50)
#     print("embedding anchor sentences")
#     anchor_embeddings = np.zeros((len(anchors), EMBEDDING_SIZE))
#     anchor_scores = np.zeros((len(anchors), ))
#     for i, anchor_sentence in enumerate(anchors):
#         resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=anchor_sentence)
#         anchor_embeddings[i, :] = resp['embedding']
        # anchor_scores[i] = project(test_embeddings[i, :], model)

print("=" * 50)
print("Embedding TEST sentences")
test_embeddings = embed_list(TEST_SET, EMBEDDING_MODEL, EMBEDDING_SIZE)
test_labels = TEST_SET['labels']
test_labels /= np.max(test_labels) # normalization

# for i, test_sentence in enumerate(TEST_SET):
#     resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=test_sentence['text'])
#     test_embeddings[i, :] = resp['embedding']
#     test_labels[i] = test_sentence['label']

##############################
# FIT REGRESSION
##############################
print("#" * 60)
print("1. Fitting via RIDGE REGRESSION..")
# 1. Fit the axis via ridge regression (closed-form, no iterative "training" loop).
#    RidgeCV searches over alpha (regularization strength) using internal CV,
#    which matters a lot here since N << 768 (see the underdetermined-system point).
alphas = np.logspace(-5, 0, 30)
ridge_model = RidgeCV(alphas=alphas, store_cv_results=True)

# alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, cv=None, gcv_mode=None, store_cv_results=False, alpha_per_target=False
ridge_model.fit(anchor_embeddings, anchor_labels)

axis = ridge_model.coef_        # shape (768,) - this is your "axis" vector w
intercept = ridge_model.intercept_
print("Chosen alpha:", ridge_model.alpha_)

# 2. Sanity-check: does the fit actually track the labels, or is it just
#    memorizing (which is trivially possible when N << 768)?
#    Leave-one-out cross-validated predictions are the honest check.
loo_preds = cross_val_predict(RidgeCV(alphas=alphas), X, y, cv=LeaveOneOut())
rho, p = spearmanr(y, loo_preds)
print(f"Leave-one-out Spearman correlation: {rho:.3f} (p={p:.4f})")
# If this is weak/non-significant, the in-sample fit is likely just
# overfitting noise in 768 dims - don't trust the axis yet; get more N
# or shrink dimensionality first (see step 4).

print("-" * 50)
print("EVALUATION ON TEST SET")
print("Ridge Regression")
RR_test_scores, RR_test_errors, RR_test_min, RR_test_max = evaluate(test_embeddings, test_labels, ridge_model, print_results=True)

# print(f"Alignment score: {score}\nLabel score: {test_sentence['label']}")


# 4. Optional but recommended given N << 768: reduce dimensionality first,
#    then fit ridge in the reduced space (fixes the underdetermined-system issue).
pca = PCA(n_components=min(PCA_COMPONENTS, len(X) - 1))
X_reduced = pca.fit_transform(anchor_embeddings)   # fit PCA on your anchors (or a larger reference corpus)
model_reduced = RidgeCV(alphas=alphas).fit(X_reduced, anchor_labels)

print("-" * 50)
print(f"PCA + Ridge Regression (PCA with {PCA_COMPONENTS} components)")
PCA_RR_test_scores, PCA_RR_test_errors, PCA_RR_test_min, PCA_RR_test_max = evaluate(test_embeddings, test_labels, model_reduced, print_results=True)

# errors = np.zeros((len(TEST_SET), 1))
# min_val = 1
# max_val = 0
# for i, test_sentence in enumerate(TEST_SET):
#     score = project_reduced(test_embeddings[i, :], pca, model_reduced) 
#     if score < min_val:
#         min_val = score 
#     if score > max_val:
#         max_val = score
#     errors[i] = np.abs(score - test_sentence['label'])
#     # print(f"Alignment score: {score}\nLabel score: {test_sentence['label']}")

# print(f"with fitted line after PCA on the embeddings from {EMBEDDING_MODEL}:")
# print(f"    MAE: {(np.mean(errors)):.03f}")
# print(f"    MSE: {(np.mean(np.square(errors))):.03f}")
# print(f"    RMSE: {(np.mean(np.sqrt(np.square(errors)))):.03f}")
# print(f"    min_val: {min_val:.03f}")
# print(f"    max_val: {max_val:.03f}")




print("-" * 50)
print("TRAIN vs TEST set metrics")
print("-" * 50)
RR_anchors_scores, RR_anchor_errors, RR_anchor_min, RR_anchor_max = evaluate(anchor_embeddings, anchor_labels, ridge_model)
# RR_anchors_scores = np.zeros((len(TEST_SET), 1))
# for i, anchor_emb in enumerate(X):
#     RR_anchors_scores[i] = project(anchor_emb, ridge_model)
# RR_test_scores = np.zeros((len(TEST_SET), 1))
# for i, test_embd in enumerate(test_embeddings):
#     RR_test_scores[i] = project(test_embd, ridge_model)

RR_train_metrics = evaluate_probe(anchor_labels, RR_anchors_scores, "Anchor (Train)")
RR_test_metrics  = evaluate_probe(test_labels,  RR_test_scores,  "Test")

#
# EVALUATE PROBE

print("=" * 50)

# ── Assume you already have these from your probe ──
# y_true_test:  list/array of true labels for test sentences
# y_pred_test:  list/array of predicted labels for test sentences
# y_true_train: list/array of true labels for anchor sentences
# y_pred_train: list/array of predicted labels for anchor sentences


    # breakpoint()
    # print(y.shape, anchor_embeddings.shape)
    # print(test_labels, test_embeddings.shape)
    # Evaluate on both sets

    # # Quick overfitting check
    # if train_metrics["r2"] - test_metrics["r2"] > 0.2:
    #     print("\n⚠️  Large gap between train and test R² — possible overfitting.")
    # elif test_metrics["r2"] < 0.3:
    #     print("\n⚠️  Low test R² — probe may be underfitting or embeddings don't encode this axis well.")
    # else:
    #     print("\n✅  Train/test gap is reasonable — probe appears to generalize.")




# X: (N, 768) anchor embeddings, y: (N,) labels in [0, 1]

# ---------- 1. Polynomial features + Ridge ----------
# Must reduce dimensionality first - degree-2 features on 768 raw dims is
# combinatorially enormous (768 choose 2 ≈ 295k new features from 15 points).
# n_components = min(15, len(X) - 1)   # keep well below N to avoid a new singular system
pca = PCA(n_components=PCA_COMPONENTS)
X_reduced = pca.fit_transform(test_embeddings)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_reduced)   # now includes pairwise interaction terms

alphas = np.logspace(-2, 4, 30)
poly_model = RidgeCV(alphas=alphas)
poly_model.fit(X_poly, test_labels)

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

poly_preds = loo_poly_eval(X, test_labels, PCA_COMPONENTS)
rho_poly, _ = spearmanr(test_labels, poly_preds)
print(f"Polynomial+Ridge LOO Spearman: {rho_poly:.3f}")
print("-" * 50)
print("EVALUATION ON TEST SET")
print("PCA + Polynomial + Ridge")
PPR_test_scores, PPR_test_errors, PPR_test_min, PPR_test_max = evaluate(test_embeddings, test_labels, poly_model, print_results=True)

# print("=" * 50)
# print("EVALUATION ON TEST SET\nRidge Regression projection on TEST SET")
# errors = np.zeros((len(TEST_SET), 1))
# min_val = 1
# max_val = 0
# test_scores = np.zeros((len(TEST_SET), ))
# for i, test_sentence in enumerate(TEST_SET):
#     score = project(test_embeddings[i, :], ridge_model)
#     test_scores[i] = score
#     if score < min_val:
#         min_val = score 
#     if score > max_val:
#         max_val = score
#     errors[i] = np.abs(score - test_sentence['label'])

# print(f"Alignment score: {score}\nLabel score: {test_sentence['label']}")

# print(f"with fitted line on the embeddings from {EMBEDDING_MODEL}:")
# print(f"    MAE: {(np.mean(errors)):.03f}")
# print(f"    MSE: {(np.mean(np.square(errors))):.03f}")
# print(f"    RMSE: {(np.mean(np.sqrt(np.square(errors)))):.03f}")
# print(f"    min_val: {min_val:.03f}")
# print(f"    max_val: {max_val:.03f}")

# order = np.argsort(X_poly)
# plt.figure()
# plt.title("Projection using Polynomial + Ridge")
# plt.plot(X_poly[order], y[order], 'o', label='true labels')
# plt.plot(X_poly[order], poly_model.predict(X_poly[order]), '-', label='isotonic fit')
# plt.xlabel('raw linear axis projection'); plt.ylabel('label'); plt.legend()
# plt.show()

breakpoint()

# ---------- 2. Isotonic recalibration on your existing linear axis ----------
# Step A: fit your original linear axis (as before)
linear_model = RidgeCV(alphas=alphas).fit(X, y)
raw_projection = linear_model.predict(X)   # 1-D scores from the linear axis

# Step B: fit a monotonic curve from raw_projection -> true label
iso = IsotonicRegression(out_of_bounds='clip')  # 'clip' handles new docs outside training range
iso.fit(raw_projection, y)

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

iso_preds = loo_isotonic_eval(X, y)
rho_iso, _ = spearmanr(y, iso_preds)
print(f"Isotonic-recalibrated LOO Spearman: {rho_iso:.3f}")

# To score a new document:
def score_isotonic(x_doc, linear_model, iso):
    raw = linear_model.predict(x_doc.reshape(1, -1))
    return iso.predict(raw)[0]

# Plot the recalibration curve - useful to *see* whether it's S-shaped
# (confirming edge-compression was a real nonlinearity, not just noise)
order = np.argsort(raw_projection)
plt.figure()
plt.title("Projection using Isotonic Recalibration")
plt.plot(raw_projection[order], y[order], 'o', label='true labels')
plt.plot(raw_projection[order], iso.predict(raw_projection[order]), '-', label='isotonic fit')
plt.xlabel('raw linear axis projection'); plt.ylabel('label'); plt.legend()
plt.show()


# ---------- 4. Gaussian Process Regression ----------
# Reduce dimensionality first - GP kernels degrade in very high-dim spaces
# (distances become less informative - "concentration of measure").
n_components_gp = min(15, len(X) - 1)
pca_gp = PCA(n_components=n_components_gp)
X_gp = pca_gp.fit_transform(X)

# Kernel: constant * RBF (smooth variation) + white noise (label/embedding noise)
kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
         + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))

gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
gp.fit(X_gp, y)

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

gp_preds, gp_stds = loo_gp_eval(X, y, n_components_gp)
rho_gp, _ = spearmanr(y, gp_preds)
print(f"GP LOO Spearman: {rho_gp:.3f}")
print(f"Mean LOO predictive std: {gp_stds.mean():.3f}")  # rough sense of typical uncertainty

# Scoring a new document, WITH uncertainty:
def score_gp(x_doc, pca_gp, gp):
    x_reduced = pca_gp.transform(x_doc.reshape(1, -1))
    mean, std = gp.predict(x_reduced, return_std=True)
    return mean[0], std[0]

# score, uncertainty = score_gp(new_doc_embedding, pca_gp, gp)
# print(f"Alignment score: {score:.3f} +/- {uncertainty:.3f}")

# Plotting predictions with uncertainty bands (sorted by predicted score)
order = np.argsort(gp_preds)
x_axis = np.arange(len(y))
plt.figure()
plt.title("Projection using Gaussian Process Regression")
plt.errorbar(x_axis, gp_preds[order], yerr=gp_stds[order], fmt='o', label='GP prediction ± std')
plt.plot(x_axis, y[order], 'k*', label='true label', markersize=10)
plt.xlabel('anchor sentence (sorted by predicted score)'); plt.ylabel('alignment score')
plt.legend(); plt.show()

# # For real documents: flag any with std well above what you saw on LOO anchors -
# # that means the document sits in a region of embedding space far from anything
# # you trained on, and the score should be treated with more skepticism.
