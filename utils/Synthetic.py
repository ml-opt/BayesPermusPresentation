import numpy as np
import itertools

def calculate_hist(rankings):
  permus = []
  count = []
  m = len(rankings)

  def equals(pi, eta):
    for x, y in zip(pi, eta):
      if x != y:
        return False
    return len(pi) == len(eta)

  def isin(pi, list):
    for eta in list:
      if equals(pi, eta):
        return True
    return False

  def indexOf(arr, elem):
    for i, val in enumerate(arr):
      if val == elem:
        return i
    return -1

  def kendall(pi, eta):
    pairs = itertools.combinations(set(pi + eta), 2)
    distance = 0
    for x, y in pairs:
        a = indexOf(pi, x) - indexOf(pi, y)
        b = indexOf(eta, x) - indexOf(eta, y)
        if a * b < 0:
            distance += 1
    return distance

  for i, pi in enumerate(rankings):
    c = 1
    if not isin(pi, permus):
      for j in range(i + 1, m):
        if equals(pi, rankings[j]):
          c += 1
    
    permus.append(pi)
    count.append(c)

    mode_idx = np.argmax(count)
    mode = permus[mode_idx]

    n = 4
    hist = []

    for pi in rankings:
      hist.append(kendall(list(pi), list(mode)))

  return hist

def plot_hist(max_distance, hist, ax, title):
  hist = np.array(hist)
  count = []

  for d in range(0, max_distance + 1):
    count.append((hist == d).sum())

  ax.bar(range(max_distance + 1), count, color='gray')
  ax.set_title(title)
  ax.set_xlabel('Distance to mode', fontsize=24)

def synthetic(num_instances, mean, std):
  assert(len(mean) == len(std))

  num_algorithms = len(mean)
  scores = np.empty((num_instances, num_algorithms))
  for i in range(num_instances):
    for j in range(num_algorithms):
      scores[i, j] = np.random.normal(mean[j], std[j])
  return scores