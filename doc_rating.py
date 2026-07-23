import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.stats import spearmanr
import ollama

# X: shape (N, 768) - embeddings of your N graded sentences
anchor_sentences = [
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
]

EMBEDDING_MODEL = 'embeddinggemma'
EMBEDDING_SIZE = 768

X = np.zeros((len(anchor_sentences), EMBEDDING_SIZE))

for i, sentence in enumerate(anchor_sentences):
    resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=sentence)
    X[i, :] = resp["embedding"]

# y: shape (N,)      - their Likert labels, e.g. [1, 1, 2, 3, 4, 5, 5, ...]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# 1. Fit the axis via ridge regression (closed-form, no iterative "training" loop).
#    RidgeCV searches over alpha (regularization strength) using internal CV,
#    which matters a lot here since N << 768 (see the underdetermined-system point).
alphas = np.logspace(-2, 4, 30)
model = RidgeCV(alphas=alphas, store_cv_results=True)
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

score = project(new_doc_embedding, model)
print("Alignment score:", score)

# 4. Optional but recommended given N << 768: reduce dimensionality first,
#    then fit ridge in the reduced space (fixes the underdetermined-system issue).
from sklearn.decomposition import PCA
pca = PCA(n_components=min(30, len(X) - 1))
X_reduced = pca.fit_transform(X)   # fit PCA on your anchors (or a larger reference corpus)
model_reduced = RidgeCV(alphas=alphas).fit(X_reduced, y)

def project_reduced(x_doc, pca, model):
    return np.dot(pca.transform(x_doc.reshape(1, -1)), model.coef_)[0] + model.intercept_