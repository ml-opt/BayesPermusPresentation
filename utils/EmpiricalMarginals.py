import numpy as np

def empirical_top_ranking_probs(orderings):
  n, m =orderings.shape
  probs = []

  for i in range(m):
    p_empirical = 0
    for order in orderings:
      if order[0] == i + 1:
        p_empirical += 1
    probs.append(p_empirical / n)

  return probs

def empirical_better_than(orderings):
  def indexOf(arr, elem):
    for i, val in enumerate(arr):
      if val == elem:
        return i
    return -1

  n, m =orderings.shape
  probs = np.zeros((n, m))

  for i in range(m):
    for j in range(m):
      if i != j:
        p_empirical = 0
        for order in orderings:
          if indexOf(order, i + 1) < indexOf(order, j + 1):
            p_empirical += 1
        probs[i, j] = p_empirical / n

  return probs

def empirical_top_k(orderings):
  def indexOf(arr, elem):
    for i, val in enumerate(arr):
      if val == elem:
        return i
    return -1

  n, m =orderings.shape
  probs = np.zeros((n, m))

  for i in range(m):
    for j in range(m):
        p_empirical = 0
        for order in orderings:
          if indexOf(order, j + 1) <= i:
            p_empirical += 1
        probs[i, j] = p_empirical / n

  return probs