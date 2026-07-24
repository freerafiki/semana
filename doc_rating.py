import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.stats import spearmanr
import ollama

# import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

EMBEDDING_MODEL = 'nomic-embed-text-v2-moe:latest' #embeddinggemma:latest' #qwen3-embedding:0.6b' #'qwen3-embedding:0.6b' # 'nomic-embed-text-v2-moe:latest'
EMBEDDING_SIZE = 768 #1024 #1024 #768
EVALUATE_PROBE = True


# X: shape (N, 768) - embeddings of your N graded sentences
anchor_sentences = [
    "No licenses, permits, or price controls exist anywhere; anyone may buy, sell, or start a business without government approval.",
    "Trade, wages, and prices are set entirely by voluntary exchange, with the state's role limited to enforcing private contracts.",
    "Businesses should operate without government interference, letting supply and demand determine prices naturally.", # 1 Pure free market
    "Regulations stifle innovation; the market self-corrects through competition and consumer choice.",                 # 2 Minimal oversight
    "The role of government is limited to enforcing contracts and protecting property rights only.",                    # 3 Light touch
    "Antitrust enforcement prevents monopolies while preserving entrepreneurial freedom elsewhere.",                    # 4 Baseline fairness
    "Basic labor standards ensure fair wages without constraining business flexibility significantly.",                 # 5 Worker protections
    "Pollution controls prevent externalities while allowing industries substantial operational autonomy.",             # 6 Environmental baseline
    "Consumer safety regulations coexist with deregulation in sectors proven to be self-monitoring.",                   # 7 Moderate balance
    "Market mechanisms guide most decisions, with strategic government intervention in natural monopolies.",            # 8 Mixed economy center
    "Essential services like healthcare require oversight to guarantee universal access alongside private options.",    # 9 Social safeguards
    "Financial institutions need transparency requirements to prevent systemic risks while maintaining competitiveness.", # 10 Balanced oversight
    "Price caps on essential goods protect vulnerable populations during market volatility.",                           # 11 Stronger framework
    "Government sets industry priorities and production targets while allowing some private enterprise.",               # 12 Active planning
    "State ownership dominates key sectors, with private businesses operating under strict licensing regimes.",         # 13 Heavy direction
    "Central planners dictate pricing and output quotas, with profit motives severely constrained.",                    # 14 Command elements
    "All economic activity is centrally planned; markets serve purely as distribution mechanisms under state authority.", # 15 Total control
    "Every price, wage, and production quota is fixed by state planners; private commerce is prohibited in all sectors.",
    "The government owns all enterprises and allocates all goods according to a central plan, with no market exchange permitted.",
]



X = np.zeros((len(anchor_sentences), EMBEDDING_SIZE))

print("=" * 50)
print(f"Embedding the sentences with {EMBEDDING_MODEL},.")
for i, sentence in enumerate(anchor_sentences):
    resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=sentence)
    X[i, :] = resp["embedding"]

# y: shape (N,)      - their Likert labels, e.g. [1, 1, 2, 3, 4, 5, 5, ...]
y = np.asarray([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 15]) / 15

print("Fitting via ridge regession..")
# 1. Fit the axis via ridge regression (closed-form, no iterative "training" loop).
#    RidgeCV searches over alpha (regularization strength) using internal CV,
#    which matters a lot here since N << 768 (see the underdetermined-system point).
alphas = np.logspace(-5, -2, 30)
model = RidgeCV(alphas=alphas, store_cv_results=True)

# alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, cv=None, gcv_mode=None, store_cv_results=False, alpha_per_target=False
model.fit(X, y)

axis = model.coef_        # shape (768,) - this is your "axis" vector w
intercept = model.intercept_
print("Chosen alpha:", model.alpha_)

# 2. Sanity-check: does the fit actually track the labels, or is it just
#    memorizing (which is trivially possible when N << 768)?
#    Leave-one-out cross-validated predictions are the honest check.
loo_preds = cross_val_predict(RidgeCV(alphas=alphas), X, y, cv=LeaveOneOut())
rho, p = spearmanr(y, loo_preds)
print(f"Leave-one-out Spearman correlation: {rho:.3f} (p={p:.4f})")
# If this is weak/non-significant, the in-sample fit is likely just
# overfitting noise in 768 dims - don't trust the axis yet; get more N
# or shrink dimensionality first (see step 4).

# 3. Project any new document embedding onto the axis - this is the
#    whole "scoring" operation, a single dot product:
def project(x_doc, model):
    return np.dot(x_doc, model.coef_) + model.intercept_

test_sentences = [
    {
        "text": "The private sector operates most efficiently when entrepreneurs are granted absolute freedom to allocate capital, set prices, and determine employment terms without bureaucratic constraints; government intervention historically distorts these natural market signals, creating artificial bottlenecks that ultimately harm consumers through inflated costs and reduced innovation, so the optimal arrangement leaves virtually all economic decisions to voluntary exchanges between informed participants in competitive environments.",
        "label": 0.05,  # Very low - strongly free market
        "style": "libertarian manifesto / polemical"
    },
    {
        "text": "A pragmatic approach to governance recognizes that neither unfettered capitalism nor comprehensive state planning delivers optimal outcomes across diverse sectors; instead, policymakers should calibrate oversight intensity according to each industry's characteristics—permitting light-touch regulation in fast-moving technology markets while maintaining stricter safety protocols in healthcare and finance—and periodically reassess these frameworks as market conditions evolve and new information emerges about which interventions actually improve social welfare without undermining productive incentives.",
        "label": 0.52,  # Medium - mixed/equilibrating
        "style": "policy paper / technocratic"
    },
    {
        "text": "Given the profound power asymmetries inherent in capitalist systems, where corporate entities routinely exploit workers, degrade communities, and externalize environmental costs onto vulnerable populations who lack bargaining leverage, the state must assert comprehensive control over pricing mechanisms, production quotas, and investment priorities to ensure that economic activity serves collective needs rather than private profit maximization; this necessitates abolishing speculative financial markets, nationalizing critical infrastructure including energy and transportation networks, and establishing democratic planning committees that coordinate all major resource allocation decisions to prioritize human flourishing over shareholder returns.",
        "label": 0.95,  # Very high - strong central planning
        "style": "radical critique / activist rhetoric"
    },
    {
        "text": "When entrepreneurs face zero regulatory burden, capital flows to its highest-value uses automatically through the price mechanism; attempts by bureaucrats to second-guess these market signals invariably produce malinvestment, shortages, and stagnant living standards.",
        "label": 0.08,
        "style": "classical economics textbook"
    },
    {
        "text": "Small businesses should enjoy considerable latitude in how they structure operations, though reasonable workplace safety requirements and basic consumer protection standards remain necessary to prevent exploitative practices that the market alone cannot reliably deter.",
        "label": 0.22,
        "style": "moderate conservative op-ed"
    },
    {
        "text": "Environmental regulations impose real costs on producers and can slow economic growth, yet some baseline pollution standards justify themselves as corrections for market failures where firms would otherwise dump toxic waste onto society without compensation.",
        "label": 0.38,
        "style": "economist blog post"
    },
    {
        "text": "The ideal regulatory regime combines competitive market dynamics in most sectors with targeted oversight of industries exhibiting natural monopoly characteristics, ensuring that utilities, telecommunications, and transport networks serve the public interest without suppressing entrepreneurial energy elsewhere.",
        "label": 0.48,
        "style": "policy think tank report"
    },
    {
        "text": "While private enterprise drives innovation effectively, pharmaceuticals, food production, and medical treatment demand stringent pre-market approval processes and ongoing monitoring because the cost of failure falls disproportionately on ordinary consumers lacking technical expertise to evaluate risks themselves.",
        "label": 0.58,
        "style": "public health advocacy statement"
    },
    {
        "text": "Financial markets have repeatedly demonstrated their inability to self-regulate without catastrophic consequences; therefore, comprehensive capital controls, strict lending standards, and continuous supervisory oversight constitute essential safeguards against systemic collapse that threatens everyone regardless of income.",
        "label": 0.72,
        "style": "progressive reformer manifesto"
    },
    {
        "text": "Private ownership persists but operates within narrow channels established by central planners who determine acceptable profit margins, priority production sectors, and wage structures based on nationally coordinated five-year development targets designed to achieve rapid industrialization.",
        "label": 0.83,
        "style": "state socialist planning document"
    },
    {
        "text": "Individual economic choices constitute acts of class collaboration with the oppressor classes; genuine liberation requires complete abolition of market relations, with all production facilities placed under workers' councils that coordinate allocation through democratic deliberation focused exclusively on meeting human needs rather than generating profits.",
        "label": 0.97,
        "style": "radical leftist theoretical work"
    },
    { "text": "Tariffs and trade barriers are lifted entirely, letting global market forces set the terms of exchange.", "label": 0.00},
    { "text": "A minimal licensing system exists only to verify professional credentials, with no other market interference.", "label": 0.07},
    { "text": "Environmental permits are required for industrial projects, but firms otherwise set their own production levels.", "label": 0.20},
    { "text": "A national minimum wage and workplace safety codes apply universally, alongside otherwise open competition.", "label": 0.28},
    { "text": "Utilities operate as regulated monopolies, while most other consumer goods markets remain unrestricted.", "label": 0.35},
    { "text": "Rent control caps are imposed in housing markets, though most other sectors set prices freely.", "label": 0.42},
    { "text": "Import quotas and production subsidies steer key industries, while smaller businesses compete with less interference.", "label": 0.55},
    { "text": "The government owns major utilities and transportation networks, leaving retail and services to private firms.", "label": 0.68},
    { "text": "Five-year plans set output targets for major industries, with private trade permitted only in minor consumer goods.", "label": 0.85},
    { "text": "Wages, prices, and production quotas are fixed by a central authority across nearly the entire economy.", "label": 0.95},
]
print("=" * 50)
print("embedding test sentences")
test_embeddings = np.zeros((len(test_sentences), EMBEDDING_SIZE))
test_labels = np.zeros((len(test_sentences), 1))
for i, test_sentence in enumerate(test_sentences):
    resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=test_sentence['text'])
    test_embeddings[i, :] = resp['embedding']
    test_labels[i] = test_sentence['label']

print("=" * 50)
print("Testing with projection on the fitted line")
errors = np.zeros((len(test_sentences), 1))
min_val = 1
max_val = 0
for i, test_sentence in enumerate(test_sentences):
    score = test_embeddings[i, :]
    if score < min_val:
        min_val = score 
    if score > max_val:
        max_val = score
    errors[i] = np.abs(score - test_sentence['label'])

print(f"Alignment score: {score}\nLabel score: {test_sentence['label']}")

print(f"with fitted line on the embeddings from {EMBEDDING_MODEL}:")
print(f"    MAE: {(np.mean(errors)):.03f}")
print(f"    MSE: {(np.square(errors)):.03f}")
print(f"    RMSE: {(np.sqrt(np.square(errors))):.03f}")
print(f"    min_val: {min_val:.03f}")
print(f"    max_val: {max_val:.03f}")


# 4. Optional but recommended given N << 768: reduce dimensionality first,
#    then fit ridge in the reduced space (fixes the underdetermined-system issue).
pca = PCA(n_components=min(15, len(X) - 1))
X_reduced = pca.fit_transform(X)   # fit PCA on your anchors (or a larger reference corpus)
model_reduced = RidgeCV(alphas=alphas).fit(X_reduced, y)

def project_reduced(x_doc, pca, model):
    return np.dot(pca.transform(x_doc.reshape(1, -1)), model.coef_)[0] + model.intercept_

print("=" * 50)
print("Testing with projection on the fitted line after using PCA")
errors = np.zeros((len(test_sentences), 1))
min_val = 1
max_val = 0
for i, test_sentence in enumerate(test_sentences):
    score = test_embeddings[i, :]    
    if score < min_val:
        min_val = score 
    if score > max_val:
        max_val = score
    errors[i] = np.abs(score - test_sentence['label'])
    print(f"Alignment score: {score}\nLabel score: {test_sentence['label']}")

print(f"with fitted line after PCA on the embeddings from {EMBEDDING_MODEL}:")
print(f"    MAE: {(np.mean(errors)):.03f}")
print(f"    MSE: {(np.square(errors)):.03f}")
print(f"    RMSE: {(np.sqrt(np.square(errors))):.03f}")
print(f"    min_val: {min_val:.03f}")
print(f"    max_val: {max_val:.03f}")

print("=" * 50)


#
# EVALUATE PROBE



# ── Assume you already have these from your probe ──
# y_true_test:  list/array of true labels for test sentences
# y_pred_test:  list/array of predicted labels for test sentences
# y_true_train: list/array of true labels for anchor sentences
# y_pred_train: list/array of predicted labels for anchor sentences

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

if EVALUATE_PROBE == True:
    print("=" * 50)
    print("embedding anchor sentences")
    anchor_embeddings = np.zeros((len(anchor_sentences), EMBEDDING_SIZE))
    for i, anchor_sentence in enumerate(anchor_sentences):
        resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=anchor_sentence)
        anchor_embeddings[i, :] = resp['embedding']
    # Evaluate on both sets
    train_metrics = evaluate_probe(y, anchor_embeddings, "Anchor (Train)")
    test_metrics  = evaluate_probe(test_labels,  test_embeddings,  "Test")

    # Quick overfitting check
    if train_metrics["r2"] - test_metrics["r2"] > 0.2:
        print("\n⚠️  Large gap between train and test R² — possible overfitting.")
    elif test_metrics["r2"] < 0.3:
        print("\n⚠️  Low test R² — probe may be underfitting or embeddings don't encode this axis well.")
    else:
        print("\n✅  Train/test gap is reasonable — probe appears to generalize.")






# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import Ridge, RidgeCV
# from sklearn.isotonic import IsotonicRegression
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
# from sklearn.model_selection import LeaveOneOut, cross_val_predict
# from scipy.stats import spearmanr
# import matplotlib.pyplot as plt

# # X: (N, 768) anchor embeddings, y: (N,) labels in [0, 1]

# # ---------- 1. Polynomial features + Ridge ----------
# # Must reduce dimensionality first - degree-2 features on 768 raw dims is
# # combinatorially enormous (768 choose 2 ≈ 295k new features from 15 points).
# n_components = min(15, len(X) - 1)   # keep well below N to avoid a new singular system
# pca = PCA(n_components=n_components)
# X_reduced = pca.fit_transform(X)

# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_reduced)   # now includes pairwise interaction terms

# alphas = np.logspace(-2, 4, 30)
# poly_model = RidgeCV(alphas=alphas)
# poly_model.fit(X_poly, y)

# # Nested LOO evaluation (redo PCA+poly+ridge per fold to avoid leakage)
# def loo_poly_eval(X, y, n_components):
#     preds = np.zeros(len(y))
#     loo = LeaveOneOut()
#     for train_idx, test_idx in loo.split(X):
#         pca_f = PCA(n_components=n_components).fit(X[train_idx])
#         Xtr, Xte = pca_f.transform(X[train_idx]), pca_f.transform(X[test_idx])
#         poly_f = PolynomialFeatures(degree=2, include_bias=False).fit(Xtr)
#         Xtr_p, Xte_p = poly_f.transform(Xtr), poly_f.transform(Xte)
#         m = RidgeCV(alphas=alphas).fit(Xtr_p, y[train_idx])
#         preds[test_idx] = m.predict(Xte_p)
#     return preds

# poly_preds = loo_poly_eval(X, y, n_components)
# rho_poly, _ = spearmanr(y, poly_preds)
# print(f"Polynomial+Ridge LOO Spearman: {rho_poly:.3f}")


# # ---------- 2. Isotonic recalibration on your existing linear axis ----------
# # Step A: fit your original linear axis (as before)
# linear_model = RidgeCV(alphas=alphas).fit(X, y)
# raw_projection = linear_model.predict(X)   # 1-D scores from the linear axis

# # Step B: fit a monotonic curve from raw_projection -> true label
# iso = IsotonicRegression(out_of_bounds='clip')  # 'clip' handles new docs outside training range
# iso.fit(raw_projection, y)

# # Nested LOO evaluation (refit both stages per fold)
# def loo_isotonic_eval(X, y):
#     preds = np.zeros(len(y))
#     loo = LeaveOneOut()
#     for train_idx, test_idx in loo.split(X):
#         lin = RidgeCV(alphas=alphas).fit(X[train_idx], y[train_idx])
#         raw_tr = lin.predict(X[train_idx])
#         raw_te = lin.predict(X[test_idx])
#         iso_f = IsotonicRegression(out_of_bounds='clip').fit(raw_tr, y[train_idx])
#         preds[test_idx] = iso_f.predict(raw_te)
#     return preds

# iso_preds = loo_isotonic_eval(X, y)
# rho_iso, _ = spearmanr(y, iso_preds)
# print(f"Isotonic-recalibrated LOO Spearman: {rho_iso:.3f}")

# # To score a new document:
# def score_isotonic(x_doc, linear_model, iso):
#     raw = linear_model.predict(x_doc.reshape(1, -1))
#     return iso.predict(raw)[0]

# # Plot the recalibration curve - useful to *see* whether it's S-shaped
# # (confirming edge-compression was a real nonlinearity, not just noise)
# order = np.argsort(raw_projection)
# plt.plot(raw_projection[order], y[order], 'o', label='true labels')
# plt.plot(raw_projection[order], iso.predict(raw_projection[order]), '-', label='isotonic fit')
# plt.xlabel('raw linear axis projection'); plt.ylabel('label'); plt.legend()
# plt.show()


# # ---------- 4. Gaussian Process Regression ----------
# # Reduce dimensionality first - GP kernels degrade in very high-dim spaces
# # (distances become less informative - "concentration of measure").
# n_components_gp = min(15, len(X) - 1)
# pca_gp = PCA(n_components=n_components_gp)
# X_gp = pca_gp.fit_transform(X)

# # Kernel: constant * RBF (smooth variation) + white noise (label/embedding noise)
# kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
#          + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))

# gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
# gp.fit(X_gp, y)

# # Nested LOO evaluation
# def loo_gp_eval(X, y, n_components):
#     preds = np.zeros(len(y))
#     stds = np.zeros(len(y))
#     loo = LeaveOneOut()
#     for train_idx, test_idx in loo.split(X):
#         pca_f = PCA(n_components=n_components).fit(X[train_idx])
#         Xtr, Xte = pca_f.transform(X[train_idx]), pca_f.transform(X[test_idx])
#         k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-2, (1e-5, 1e1))
#         m = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=10)
#         m.fit(Xtr, y[train_idx])
#         pred, std = m.predict(Xte, return_std=True)
#         preds[test_idx] = pred
#         stds[test_idx] = std
#     return preds, stds

# gp_preds, gp_stds = loo_gp_eval(X, y, n_components_gp)
# rho_gp, _ = spearmanr(y, gp_preds)
# print(f"GP LOO Spearman: {rho_gp:.3f}")
# print(f"Mean LOO predictive std: {gp_stds.mean():.3f}")  # rough sense of typical uncertainty

# # Scoring a new document, WITH uncertainty:
# def score_gp(x_doc, pca_gp, gp):
#     x_reduced = pca_gp.transform(x_doc.reshape(1, -1))
#     mean, std = gp.predict(x_reduced, return_std=True)
#     return mean[0], std[0]

# score, uncertainty = score_gp(new_doc_embedding, pca_gp, gp)
# print(f"Alignment score: {score:.3f} +/- {uncertainty:.3f}")

# # Plotting predictions with uncertainty bands (sorted by predicted score)
# order = np.argsort(gp_preds)
# x_axis = np.arange(len(y))
# plt.errorbar(x_axis, gp_preds[order], yerr=gp_stds[order], fmt='o', label='GP prediction ± std')
# plt.plot(x_axis, y[order], 'k*', label='true label', markersize=10)
# plt.xlabel('anchor sentence (sorted by predicted score)'); plt.ylabel('alignment score')
# plt.legend(); plt.show()

# # For real documents: flag any with std well above what you saw on LOO anchors -
# # that means the document sits in a region of embedding space far from anything
# # you trained on, and the score should be treated with more skepticism.
