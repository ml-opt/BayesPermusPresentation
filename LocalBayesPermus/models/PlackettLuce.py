from .Base import Base

import numpy as np
import pystan

pld_code = """
data {
    int<lower=1> num_permus;
    int<lower=2> num_algorithms;
    int permus [num_permus, num_algorithms];
    vector[num_permus] weights;
    vector[num_algorithms] alpha;
}

parameters {
    simplex[num_algorithms] ratings;
}
 
transformed parameters {
  real loglik;
  real rest;
  
  loglik = 0;
  for (s in 1:num_permus){
    for (i in 1:(num_algorithms - 1)) {
      rest = 0;

      for (j in i:num_algorithms) {
        rest = rest + ratings[permus[s, j]];
      }

      loglik = loglik + log(weights[s] * ratings[permus[s, i]] / rest);
    }
  }
}
 
model {
    ratings ~ dirichlet(alpha);
    target += loglik;
}
"""

class PlackettLuceDirichlet(Base):
  """ The Plackett-Luce model using a Dirichlet prior.
  """

  def __init__(self, alpha, seed=1, num_chains=1, num_samples=2000):
    Base.__init__(self, stan_model=pld_code, 
                  seed=seed, num_chains=num_chains, num_samples=num_samples)
    self.alpha = alpha    

  def get_model_data(self, permus, weights=None):
    num_permus, num_algorithms = permus.shape

    if weights == None:
      weights = [1 for _ in range(num_permus)]

    model_data = {'num_permus': num_permus,
                  'num_algorithms': num_algorithms,
                  'permus': permus,
                  'weights': weights,
                  'alpha': self.alpha}
    return model_data

  def calculate_permu_prob(self, permu, params):
    num_algorithms = len(permu)
    ratings = params
    prob = 1

    for i in range(num_algorithms):
      denominator = np.sum([ratings[permu[j]] for j in range(i, num_algorithms)])
      prob *= ratings[permu[i]] / denominator

    return prob

  def calculate_top_ranking_probs(self, permus, weights=None):
    """ Calculate the probability that each algorithm to be the best algorithm.

    Parameters:
    ------------

    permus : ndarray 
      Array of num_permus rows representing a permutation and num_algorithms columns 
      representing  the ranking of each algorithm.

    Returns
    --------

    probs : ndarray
      Array of size (num_posterior_samples, num_algorithms) where the 
      first dimension refers to the number of posterior samples and the second dimension
      represent each algorithm. For example, probs[10, 0] is the probability that the algorithm with
      the index 0 is the best algorithm according to the posterior sample with index 10.

    """

    samples = self.sample_posterior(permus, weights)
    num_permus, num_algorithms = permus.shape
    num_posterior_samples = len(samples)
    probs = np.zeros((num_posterior_samples, num_algorithms))

    for i, sample in enumerate(samples):
      for j, theta in enumerate(sample):
        probs[i, j] = theta

    return probs


plg_code = """
data {
    int<lower=1> num_permus;
    int<lower=2> num_algorithms;
    int permus [num_permus, num_algorithms];
    vector[num_permus] weights;
    real alpha; 
    real beta;
}

parameters {
    simplex[num_algorithms] ratings;
}
 
transformed parameters {
  real loglik;
  real rest;
  
  loglik = 0;
  for (s in 1:num_permus){
    for (i in 1:(num_algorithms - 1)) {
      rest = 0;

      for (j in i:num_algorithms) {
        rest = rest + ratings[permus[s, j]];
      }

      loglik = loglik + log(weights[s] * ratings[permus[s, i]] / rest);
    }
  }
}
 
model {
    for (i in 1:num_algorithms) {
        ratings[i] ~ gamma(alpha, beta);
    }

    target += loglik;
}
"""

class PlackettLuceGamma(Base):
  """ The Plackett-Luce model using a Gamma prior.
  """

  def __init__(self, alpha, beta, seed=1, num_chains=1, num_samples=2000):
    Base.__init__(self, stan_model=plg_code, 
                  seed=seed, num_chains=num_chains, num_samples=num_samples)
    self.alpha = alpha
    self.beta = beta  

  def get_model_data(self, permus, weights):
    num_permus, num_algorithms = permus.shape

    if weights == None:
      weights = [1 for _ in range(num_permus)]

    model_data = {'num_permus': num_permus,
                  'num_algorithms': num_algorithms,
                  'permus': permus,
                  'weights': weights,
                  'alpha': self.alpha,
                  'beta': self.beta}
    return model_data

  def calculate_permu_prob(self, permu, params):
    num_algorithms = len(permu)
    ratings = params
    prob = 1

    for i in range(num_algorithms):
      denominator = np.sum([ratings[permu[j]] for j in range(i, num_algorithms)])
      prob *= ratings[permu[i]] / denominator

    return prob

  def calculate_top_ranking_probs(self, permus, weights=None):
    """ Calculate the probability that each algorithm to be the best algorithm.

    Parameters:
    ------------

    permus : ndarray 
      Array of num_permus rows representing a permutation and num_algorithms columns 
      representing  the ranking of each algorithm.

    Returns
    --------

    probs : ndarray
      Array of size (num_posterior_samples, num_algorithms) where the 
      first dimension refers to the number of posterior samples and the second dimension
      represent each algorithm. For example, probs[10, 0] is the probability that the algorithm with
      the index 0 is the best algorithm according to the posterior sample with index 10.

    """

    samples = self.sample_posterior(permus, weights)
    num_permus, num_algorithms = permus.shape
    num_posterior_samples = len(samples)
    probs = np.zeros((num_posterior_samples, num_algorithms))

    for i, sample in enumerate(samples):
      for j, theta in enumerate(sample):
        probs[i, j] = theta

    return probs