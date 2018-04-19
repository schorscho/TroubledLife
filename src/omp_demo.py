import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_sparse_coded_signal
from sklearn.metrics import mean_squared_error


n = 10  # n_features
m = 50  # n_components
n_nonzero_coefs = 4

# For demo purposes create a consistent example, to be replaced with own case
b, A, x = make_sparse_coded_signal(n_samples=1,
                                   n_components=m,
                                   n_features=n,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)

print(A.shape, x.shape, b.shape)

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)

# "fit" to dataset - the typical scikit-learn API style
omp.fit(A, b)

# the caclulated sparse x
x_pred = omp.coef_

# of non-zero elements
print(len(x_pred.nonzero()[0]))

# l2-norm of Ax - b
l2 = mean_squared_error(b, np.matmul(A, x))

print(l2)

