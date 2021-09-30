import itertools
import numpy 

def WCSS(samples):
  # Compute 'Within Sum of Squares' of a cluster
  SumOSquares = []
  mean = np.mean(samples, axis=0)
  for point in samples:
    SumOSquares.append(np.linalg.norm(mean-point))
  return np.mean(SumOSquares)

def BCSS(group_1, group_2):
  # Compute 'Between Sum of Squares' between two clusters
  mean1 = np.mean(group_1, axis=0)
  mean2 = np.mean(group_2, axis=0)
  return np.linalg.norm(mean1-mean2)

def mean_SumSquares(samples):
  """
  As input mean_SumSquares receives a list of the latent representations of the images.
  The data must be in sorted order:
    [1008 representations of object1, 1008 representations of object2,...]
  """
  length = len(samples)
  amount_of_objects=12
  views_per_object = int(length/amount_of_objects)

  # compute 'within cluster sum of squares' for each cluster
  wcss = []
  for i in range(0,12):
    wcss.append(WCSS(samples[i*views_per_object:(i+1)*views_per_object]))

  # compute 'between cluster sum of squares' for all possible combinations of clusters
  bcss = []
  for el in itertools.combinations(range(12), 2):
    i = el[0]
    j = el[1]
    bcss.append(BCSS(samples[i*views_per_object:(i+1)*views_per_object], samples[j*views_per_object:(j+1)*views_per_object]))

  # get the mean values
  mean_bcss = np.mean(bcss)
  std_bcss = np.std(bcss)
  mean_wcss = np.mean(wcss)
  std_wcss = np.std(wcss)
  return mean_bcss, std_bcss, mean_wcss, std_wcss
