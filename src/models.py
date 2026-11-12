import numpy as np

def bias_baseline(R):
  # create a boolean mask where True indicates a known rating exists (is not NaN)
  known = ~np.isnan(R)

  # compute the global mean (mu) across all known ratings, skipping NaNs
  mu = np.nanmean(R)

  # compute the user bias as the average deviation from the global mean for each user
  b_user = np.nanmean(R - mu, axis=1)

  # compute the residual matrix by subtracting the global mean and user biases from the ratings
  residual = R - mu - b_user[:, None]

  # calculate the count of training ratings for each item
  n_item = known.sum(axis=0)

  # initialize movie biases with 0 for all items
  b_item = np.zeros(R.shape[1])

  # create a boolean mask to locate items that have at least one rating in the training matrix
  has_rating = n_item > 0

  # for movies that are rated, compute item bias by avging the residual ratings and skipping NaNs
  b_item[has_rating] = (np.nansum(residual, axis=0)[has_rating] / n_item[has_rating])

  # return the global mean, user bias array, and movie bias array
  return mu, b_user, b_item


# build a full prediction matrix from the bias baseline
def baseline_matrix(R):
  # get the global mean, user biases, and movie biases from the known ratings
  mu, b_user, b_item = bias_baseline(R)

  # b_user[:, None] is a column and b_item[None, :] is a row, so numpy broadcasting
  # adds them into a full (users x movies) matrix with a prediction for every pair
  return mu + b_user[:, None] + b_item[None, :]


# build a rank-k matrix from an svd that was already computed
def rank_k(U, s, Vt, k):
  # keep only the first k columns of U, the first k singular values, and the
  # first k rows of Vt, then multiply them back together
  return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


# predict ratings by repeatedly filling in the missing deviations with svd guesses
def iterative_svd_predict(R, k, n_iter=10):
  # boolean mask of which ratings are known
  known = ~np.isnan(R)

  # baseline prediction for every user-movie pair
  B = baseline_matrix(R)

  # deviations from the baseline at known ratings, 0 in the gaps to start
  X = np.where(known, R - B, 0.0)

  for _ in range(n_iter):
    # take the svd of the current deviation matrix and keep the rank-k part
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    L = rank_k(U, s, Vt, k)

    # keep the real deviations at known ratings and use the rank-k guesses in the gaps
    X = np.where(known, R - B, L)

  # add the baseline back so the predictions are in stars
  return L + B
