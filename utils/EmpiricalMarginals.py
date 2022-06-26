import numpy as np

def empirical_top_ranking_probs(scores):
  m, n = scores.shape
  probs = np.zeros(n)
  
  for score in scores:
    indices = np.argsort(score)
    probs[indices[0]] += 1
    
    for i in range(1, n):
      if score[indices[i - 1]] == score[indices[i]]:
        probs[indices[i]] += 1
      else:
        break
  
  return probs / np.sum(probs)

def empirical_better_than(scores):
  n, m = scores.shape
  probs = np.zeros((m, m))

  for score in scores:
    indices = np.argsort(score)

    for i in range(m):
      for j in range(i + 1, m):
        if score[i] == score[j]:
          probs[i, j] += 1
          probs[j, i] += 1
        elif score[i] < score[j]:
          probs[i, j] += 1
        elif score[i] > score[j]:
          probs[j, i] += 1

  for i in range(m):
    for j in range(i + 1, m):
      sume = probs[i, j] + probs[j, i]
      probs[i, j] /= sume
      probs[j, i] /= sume

  return probs

def empirical_top_k(scores):
  n, m = scores.shape
  probs = np.zeros((m, m))

  for score  in scores:
    indices = np.argsort(score)
    probs[:, indices[0]] += 1

    k = 0
    for i in range(1, m):
      if score[indices[i - 1]] == score[indices[i]]:
        probs[k:, indices[i]] += 1
      else:
        k += 1
        probs[k:, indices[i]] += 1

  return probs / n