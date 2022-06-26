import os.path
import math
import numpy as np
import itertools
import pystan

class Base:
  """Base class for the different Bayesian models on permutation spaces.
  This class handles the steps of performing Bayesian inference using pystan
  and sampling from the posterior distributions.

  In addition, this class provides naive implementations of the different posterior
  summaries. However, without additional information on the particular model,
  such posterior summaries are in general factorial in the numer of algorithms being
  compared. Therefore, whenever possible, derived classes should override the
  base implementations.
  """

  def __init__(self, stan_model=None, seed=1, num_chains=1, num_samples=2000):
    """ Constructor of the Base class.

    Parameters:
    -----------

    stan_model : string 
      Filename of the stan code for this particular model. Leave None in case no Stan
      model is used.

    seed : integer
      Random seed passed to Stan during posterior sampling.

    num_chains : integer
      Number of Monte Carlo chains to be performed during inference.

    num_samples : integer
      Number of posterior samples.
    """
    self.seed = seed
    self.num_chains = num_chains
    self.num_samples = num_samples
    
    if stan_model != None:
      self.posterior = pystan.StanModel(model_code=stan_model)

  def sample_posterior(self, permus, weights):
    """ Sample from the posterior distribution.

    Parameters:
    -----------

    permus : ndarray 
      Array of num_permus rows representing a permutation and num_algorithms columns 
      representing  the ranking of each algorithm.

    Returns
    --------

    samples : ndarray
      Array of size (num_posterior_samples, num_model_params) where the first dimension represents
      the number of posterior samples and the second dimension represents the number of parameter
      of the probabilistic model.

    """

    model_data = self.get_model_data(permus, weights)
    fit = self.posterior.sampling(data=model_data, chains=self.num_chains, iter=self.num_samples, seed=self.seed)
    samples = self.get_samples(fit)
    return samples


  def get_model_data(self, permus, weights):
    """
      Returns the model data.
    """

    return {"data": permus}

  def get_samples(self, data):
    """
      Returns the model samples.
    """

    return data['ratings']

  def calculate_permu_prob(self, permu, params):
    """ Computes the probability of a given permutation according to a given
    probability model defined by params. Derived classes should provide a concrete
    implementation of this method.

    Parameters:
    -----------

    permu : ndarray
      A one dimensional array representing a permutation.
    params: ndarray
      The parameters of the probability model.

    """
    return 0

  def sample_uniform_permu(self, permu):
    """ Sample an uniform random permutation and store the result in permu.

    Parameters:
    -----------

    permu : ndarray
      A one dimensional array representing a permutation.

    """

    num_algorithms = len(permu)
    for i in range(num_algorithms):
      idx = np.random.randint(i, num_algorithms)
      permu[i], permu[idx] = permu[idx], permu[i]
    return permu

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
      for permu in itertools.permutations(list(range(num_algorithms))):
        permu = np.array(permu)
        probs[i, permu[0]] += self.calculate_permu_prob(permu, sample)

    return probs

  def calculate_better_than_probs(self, permus, weights=None):
    """ Calculate the probability that each algorithm to outperforms other
    algorithms. The function computes such probability for each possible
    pair of algorithms.

    Parameters:
    ------------

    permus : ndarray 
      Array of num_permus rows representing a permutation and num_algorithms columns 
      representing  the ranking of each algorithm.

    Returns
    --------

    probs : ndarray
      Array of size (num_posterior_samples, num_algorithms, num_algorithms) where the 
      first dimension refers to the number of posterior samples, the second dimension
      represent each algorithm and the third dimension represents the algorithm which is
      outperformed. For example, probs[10, 0, 2] is the probability that the algorithm with
      the index 0 outperforms the algorithm with the index 2 according to the posterior 
      sample with index 10.

    """

    samples = self.sample_posterior(permus, weights)
    num_permus, num_algorithms = permus.shape
    num_posterior_samples = len(samples)
    probs = np.zeros((num_posterior_samples, num_algorithms, num_algorithms))

    for i, sample in enumerate(samples):
      for permu in itertools.permutations(list(range(num_algorithms))):
        permu = np.array(permu)
        prob = self.calculate_permu_prob(permu, sample)

        for j in range(num_algorithms):
          for k in range(j + 1, num_algorithms):
            probs[i, permu[j], permu[k]] += prob

    return probs

  def calculate_top_k_probs(self, permus, weights=None):
    """ Calculate the probability that each algorithm is in the top-k
    ranking. The function computes such probability for each possible
    value of k for 1 < k < num_algorithms.

    Parameters:
    ------------

    permus : ndarray 
      Array of num_permus rows representing a permutation and num_algorithms columns 
      representing  the ranking of each algorithm.

    Returns
    --------

    probs : ndarray
      Array of size (num_posterior_samples, num_algorithms, num_algorithms) where the 
      first dimension refers to the number of posterior samples, the second dimension
      represent each algorithm and the third dimension represents the value of k being
      calculated. For example, probs[10, 0, 2] is the probability to be in the top-3 ranking 
      of the algorithm with index 0 according to the posterior sample with index 10.

    """

    samples = self.sample_posterior(permus, weights=None)
    num_permus, num_algorithms = permus.shape
    num_posterior_samples = len(samples)
    probs = np.zeros((num_posterior_samples, num_algorithms, num_algorithms))

    for i, sample in enumerate(samples):
      for permu in itertools.permutations(list(range(num_algorithms))):
        permu = np.array(permu)
        prob = self.calculate_permu_prob(permu, sample)

        for j in range(num_algorithms):
          for k in range(j, num_algorithms):
            probs[i, permu[j], k] += prob

    return probs
