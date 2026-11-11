# function to get rmse using predicted value and true (actual val)
def rmse(pred, truth):
  # root mean squared error, measured in stars.
  return np.sqrt(np.mean((pred - truth)**2))

# create a function to convert the ratings into a user-item matrix
def to_matrix(df):
  # create a 2D np array of shape (943 x 1682) initialized with
  # NaN (not a number) to represent movies with missing ratings
  R = np.full((N_USERS, N_ITEMS), np.nan)

  # map ratings to the matrix:
  R[df["user"].to_numpy() - 1,df["item"].to_numpy() - 1] = df["rating"].to_numpy()
  # subtract 1 from user/item IDs to align IDs (index-1) with 0-indexing used in
  # python

  # return the new (user x movie) rating matrix
  return R

# split ratings into test and train data
def split_ratings(ratings, test_frac=0.10, seed=0):
  # create a random number generator with a fixed seed so u can ensure the same
  # random vals are generated each time
  rng = np.random.default_rng(seed)

  # make an array of bools with 0 (false) values with the same length as ratings
  is_test = np.zeros(len(ratings), dtype=bool)

  # map each user id to the total number of ratings they have provided
  user_left = ratings.groupby("user").size().to_dict()

  # map each movie id to the total number of ratings it has received
  item_left = ratings.groupby("item").size().to_dict()

  # go through each position in the random permuted sequence of ratings
  for pos in rng.permutation(len(ratings)):
    # get the row corresponding to the current position of the random index
    row = ratings.iloc[pos]
    user, item = row["user"], row["item"]

    # Only assign to the test set if we randomly meet the fraction and both user and item have other ratings
    if (rng.random() < test_frac and user_left[user] > 1 and item_left[item] > 1):
      # Mark this specific index position as belonging to the test set
      is_test[pos] = True

      # decrease the count of remaining training ratings for this user by 1
      user_left[user] -= 1

      # decrease the count of remaining training ratings for this movie by 1
      item_left[item] -= 1

  # Return two new dataframes:
  # the training set (~is_test) AKA is_test with all 0's and 1's inverted
  # and the test set (is_test)
  return ratings.iloc[~is_test].copy(), ratings.iloc[is_test].copy()
