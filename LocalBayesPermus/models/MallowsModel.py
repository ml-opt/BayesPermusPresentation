from .Base import Base

import itertools
import pandas as pd
import numpy as np
import pystan

import rpy2
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri

from rpy2.robjects.vectors import FloatVector
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
BayesMallows = importr('BayesMallows')

class MallowsModel(Base):
  """ The Mallows model.
  """

  def __init__(self, num_samples_marginals=100, seed=1, num_samples=2000):
    Base.__init__(self, seed=seed, num_chains=1, num_samples=num_samples)

  def sample_posterior(self, permus, weights=None):
    num_permus, num_algorithms = permus.shape

    # Sample from the posterior distribution.
    if weights == None:
      bmm = BayesMallows.compute_mallows(rankings=permus, nmc=self.num_samples, 
                                         seed=self.seed, metric='kendall')
    else:
      bmm = BayesMallows.compute_mallows(rankings=permus, nmc=self.num_samples, 
                                         obs_freq=FloatVector(weights), seed=self.seed, metric='kendall')
    bmm = dict(zip(bmm.names, list(bmm)))

    # Get central permutation samples stored in the `rho` key, which returns a
    # dataframe with the following info: item, cluter (not used), MC iteration 
    # and rank of the item. Each iteration is associated with a number of rows 
    # equals to `n_algorithms` conforming a single permutation in the sample.
    df_location = pd.DataFrame([list(row) for row in bmm['rho']], 
                               columns=['item', 'cluster', 'iteration', 'rank'])
    df_location['rank'] = df_location['rank'].astype(np.int)
    
    # Get dispersion sampled stored in the key `alpha` key, which returns a 
    # dataframe with the following info: cluter (not used)  MC iteration 
    # and value of the dispersion parameter.
    df_dispersion = pd.DataFrame([list(row) for row in bmm['alpha']], 
                                 columns=['cluster', 'iteration', 'value'])

    # Drop burnin iterations of the MC sample.
    df_location = df_location[df_location['iteration'] > self.num_samples / 2]
    df_dispersion = df_dispersion[df_dispersion['iteration'] > self.num_samples / 2]

    # Extract a list of location parameters from the samples of the posterior.
    # The ranks of the location parameters are ordered as follows: 
    # `[Item_1, ..., Item_m, ..., Item_1, ..., Item_m]` in the resulting
    # df_location['rank'].to_numpy() array, which is why we can reshape the 
    # array as we do here to obtain a row per central parameter of length `m`.
    locations = df_location['rank'].to_numpy().reshape(-1, num_algorithms) - 1
    dispersions = df_dispersion['value'].to_numpy()

    result = [(location, dispersion) for location, dispersion in zip(locations, dispersions)]
    return result

  def psi(self, dispersion, num_algorithms):
    psi = 1
    for i in range(1, num_algorithms):
      psi *= (1 - np.exp(-dispersion * (num_algorithms - i + 1))) / (1 - np.exp(-dispersion))
    return psi

  def indexOf(self, arr, elem):
    for i, val in enumerate(arr):
      if val == elem:
        return i
    return -1

  def kendall(self, pi, eta):
    pairs = itertools.combinations(pi, 2)
    distance = 0
    for x, y in pairs:
        a = self.indexOf(pi, x) - self.indexOf(pi, y)
        b = self.indexOf(eta, x) - self.indexOf(eta, y)
        if a * b < 0:
            distance += 1
    return distance

  def calculate_permu_prob(self, permu, params):
    num_algorithms = len(permu)
    location, dispersion = params
    prob = np.exp(-dispersion * self.kendall(np.array(location), np.array(permu))) / self.psi(dispersion, num_algorithms)
    return prob