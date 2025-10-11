# Big data / low memory problem
# Process data in chunks 

import pandas as pd

chunk_size = 100_000  # number of rows per chunk
for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
    # process the chunk
    chunk_result = chunk['column'].mean()  # example operation
    # aggregate results incrementally

# Use generators / streaming
def read_large_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

for row in read_large_file('large_dataset.csv'):
    # process row

# Use memory-efficient libraries
import dask.dataframe as dd

df = dd.read_csv('large_dataset.csv')
result = df['column'].mean().compute()  # lazy evaluation

# Use external storage / databases
import duckdb

con = duckdb.connect()
result = con.execute("SELECT AVG(column) FROM 'large_dataset.parquet'").fetchone()


# Ways to Handle Imbalanced Datasets
# Resampling Techniques
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)

# Class Weighting / Cost-Sensitive Learning
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(class_weight='balanced')
clf.fit(X_train, y_train)

# Normalize / Scale
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Proper Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ML pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

# Remove future-dependent features
# Split data before preprocessing, fit transformers on training set only
# Use chronological split for time-series data


# Correlation Matrix
import pandas as pd
corr_matrix = X.corr()

# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

# Regularization
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

"""
Overfitting can be reduced by simplifying the model, regularization, 
cross-validation, early stopping, dropout, more data, feature selection, and ensemble methods.
"""

# Group Rare Categories
threshold = 50
counts = df['Category'].value_counts()
df['Category_grouped'] = df['Category'].apply(lambda x: x if counts[x] > threshold else 'Other')


# Pruning in Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, ccp_alpha=0.01)
clf.fit(X_train, y_train)

# ROC-AUC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)


#  SVM with RBF kernel
# Kernels are functions that transform data from the original feature space to a higher-dimensional space, enabling non-linear classification
from sklearn.svm import SVC

clf = SVC(kernel='rbf', C=1.0, gamma=0.1) # RBF (Radial Basis Function) / Gaussian Kernel
clf.fit(X_train, y_train)

# Gini Impurity and Entropy are two popular splitting criteria in decision trees
# Gini Impurity measures the probability of misclassification, while Entropy measures the disorder/uncertainty.
from sklearn.tree import DecisionTreeClassifier

clf_gini = DecisionTreeClassifier(criterion='gini') # Gini Impurity
clf_entropy = DecisionTreeClassifier(criterion='entropy') # Entropy

# Standard PCA cannot effectively reduce the dimensionality of highly nonlinear datasets
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15) # Kernel PCA
X_kpca = kpca.fit_transform(X)

# TimeSeriesSplit (Scikit-Learn)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Reconstruction with PCA (Example)
from sklearn.decomposition import PCA

pca = PCA(n_components=2) # Reduce dimensionality
X_reduced = pca.fit_transform(X)

X_approx = pca.inverse_transform(X_reduced) # Approximate reconstruction


# intercept term in a regression model
# The intercept term allows the regression line to shift vertically to better fit the data.
# Without it, the model is forced through the origin, which often misrepresents relationships and increases error.
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# High variance (overfitting) occurs when the model memorizes training data.
# Handle it by simplifying the model, regularization, adding more data, removing noisy features, using ensembles, and early stopping.

# Active Learning 
Unlabeled Data Pool → Model → Select Most Informative → Label → Retrain → Repeat

"""
Ridge Regression (L2): Shrinks coefficients → reduces variance.
Lasso Regression (L1): Can remove redundant features by setting coefficients to zero.
"""

# Example Dataset
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [labels[i] for i in y]
df.head()

# PCA
from sklearn.decomposition import PCA

# Initialize PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot results
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=iris.target_names[y], palette='Set2')
plt.title('PCA projection (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)

# t-SNE (t-Distributed Stochastic Neighbor Embedding)
from sklearn.manifold import TSNE

# Initialize t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=iris.target_names[y], palette='Set2')
plt.title('t-SNE projection (2D)')
plt.show()

# UMAP (Uniform Manifold Approximation and Projection)
!pip install umap-learn

import umap.umap_ as umap

# Initialize UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)

# Plot
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=iris.target_names[y], palette='Set2')
plt.title('UMAP projection (2D)')
plt.show()

# Parallel training
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # uses all available cores
rf.fit(X_train, y_train)

# Lasso (L1) Regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# X = features, y = target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso with cross-validation to select alpha
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

# Selected features
selected_features = X.columns[lasso.coef_ != 0]
print("Selected features:", selected_features)

# After training, check model.coef_.
# Features with coefficient = 0 are considered unimportant and can be removed.

"""
# A loss function measures how well a model predicts a single training example. It quantifies the error for one sample.
# A cost function aggregates the loss over all training examples to quantify overall model performance. 
# It’s what we actually minimize during training.
"""

# Handling missing values
from sklearn.impute import SimpleImputer

# Numeric
imputer = SimpleImputer(strategy='mean')
X_num = imputer.fit_transform(X_num)

# Categorical
imputer_cat = SimpleImputer(strategy='most_frequent')
X_cat = imputer_cat.fit_transform(X_cat)

"""
StandardScaler → zero mean, unit variance, robust to outliers
MinMaxScaler → scales to a fixed range, sensitive to outliers, good for neural networks
"""

# Separate numerical and categorical features
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False)
X_cat_encoded = ohe.fit_transform(X[cat_cols])

# Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
X_cat_encoded = ord_enc.fit_transform(X[cat_cols])

# concatenate dataframes
import numpy as np

X_processed = np.hstack([
    scaler.fit_transform(X[num_cols]),  # scaled numeric
    X_cat_encoded                       # encoded categorical
])

"""
Regression models: simple, interpretable, fast, but may underfit nonlinear patterns.
Tree-based models: flexible, handle complex data, but can overfit and are less interpretable.
"""

"""
Important XGBoost hyperparameters can be grouped as: 
tree structure (max_depth, min_child_weight, gamma), 
boosting (learning_rate, n_estimators), 
regularization (alpha, lambda), and 
subsampling (subsample, colsample_bytree).
"""

# Always split into train/test before scaling or imputing, to avoid data leakage.

"""
1. Problem Definition → 2. Data Collection → 3. EDA → 4. Data Preprocessing → 
5. Feature Engineering → 6. Model Training → 7. Evaluation → 8. Interpretation → 
9. Deployment → 10. Monitoring → 11. Reporting
"""

"""
A good ML model is accurate, generalizable, interpretable, robust, efficient, 
simple, reproducible, adaptable, and balanced between bias and variance.
"""

# R-squared measures the proportion of variance explained by the model.
# Adjusted R-squared adjust r-squared for number of predictors (p) and sample size (n).

# evaluation metrices for a classification model with imbalanced data
# Matthews Correlation Coefficient (MCC)
# Cohen’s Kappa

"""
The Curse of Dimensionality means that high-dimensional feature spaces become sparse, 
distances become less meaningful, and models overfit easily, making learning and generalization harder.
"""

"""
The Bias-Variance Tradeoff is about finding the right model complexity:
Too simple → High bias, underfits
Too complex → High variance, overfits
Sweet spot → Minimum total error and best generalization
"""

"""
Error	        Occurs when	                Result
Type I (α)	    H₀ true, but rejected	    False alarm / false positive
Type II (β)	    H₀ false, but not rejected	Missed detection / false negative
"""

"""
Generative models learn how the data is generated (P(X,Y)) and can generate new samples. - Naive Bayes
Discriminative models learn how to separate classes (P(Y∣X)) and focus purely on prediction. - SVM
"""

"""
Binary crossentropy and categorical crossentropy differ in output activation, 
label interpretation, and gradient computation. Using the “wrong” one for your problem 
changes how the network learns and can lead to different performance.
"""

# One-hot encoding ensures all categories are equidistant, preventing misleading results.

"""
Wide format = one row per entity, multiple columns for variables.
Tall/long format = one row per observation/measurement, fewer columns but more rows.
"""

"""
Inductive ML = data-driven generalization. -- Learn from examples to generalize.
Deductive ML = knowledge/rule-driven reasoning. -- Use existing knowledge to infer specifics.
"""

"""
Covariance tells you the direction of a relationship, 
while correlation tells you the strength and direction of a linear relationship in a standardized, interpretable way.
"""

"""
Gradient Descent uses the full dataset for each update (stable but slow), 
whereas Stochastic Gradient Descent uses one sample at a time (faster but noisy). 
Mini-batch is a practical compromise used in most ML/DL applications.
"""

"""
K-Means is a simple, distance-based hard clustering algorithm assuming spherical clusters.
GMM is a probabilistic model that assumes data comes from a mixture of Gaussians, allowing soft assignments and elliptical clusters.
"""

"""
More data can improve performance, especially for complex models, 
but only if it’s high-quality, relevant, and the model has enough capacity to learn from it. 
Bad or redundant data won’t help, and very large datasets increase computation costs.
"""

