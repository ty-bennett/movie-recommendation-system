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
