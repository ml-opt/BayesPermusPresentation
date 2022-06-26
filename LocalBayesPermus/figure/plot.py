import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np

class Plot:
  """
    This class contains methods for ploting the different posterior summaries of interest.
  """

  def set_axis_style(self, ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

  def plot_top_ranking_probs(self, model_names, algorithm_names, probs, empirical, axs):
    """ Plot the probability that each algorithm is in the first ranking.

    This function generates several violin plots, one per algorithm being compared.

    Each plot represents in the horizontal axis the different Bayesian models under consideration,
    the vertical axis represents the probability. 

    If given, the function plots an horizontal black dashed line that represents the empirical probability.

    Parameters:
    -----------

    model_names : list of string
      A list of strings representing the different model names.

    model_names : list of string
      A list of strings representing the different algorithm names.

    probs : list of ndarray
      A list containing several matrices, one per Bayesian model under consideration.
      The inner 2-dimensional matrices are returned by calculate_top_ranking_probs().

    empirical: list of float
      A list of empirical probabilities, one entry for each algorithm being compared.

    axs : list of matplotlib axes
      A list containing matplotlib axes, one per algorithm being considered.
      In case some axis is None, the method will skip that particular plot.
    """

    assert(len(model_names) > 0)
    assert(len(model_names) == len(probs))
    assert(len(algorithm_names) == len(axs))

    num_samples, num_algorithms = probs[0].shape

    for i, ax in enumerate(axs):
      if ax != None:
        df = pd.DataFrame(columns=model_names)

        for model_name, sample in zip(model_names, probs):
          df[model_name] = sample[:,i]
        
        ax.violinplot(df.values, showmeans=True)
        ax.axhline(y=empirical[i], linestyle=':', color='black', label='Empirical')
        ax.set_title(algorithm_names[i])
        self.set_axis_style(ax, model_names)

  def plot_better_than_probs(self, algorithm_names, probs, empirical, axs):
    """ Plot the probability that each algorithm is better than other algorithm.

    This function generates several scatter plots in polar coordinates. There are several 
    scatter plots, one for each algorithm being compared. 

    Each scatter plot in polar coordinates is associated to a given algorithm represented in the
    origin of the coordinate system.

    The plot is divided in several sectors, one sector for each of the other algorithms.
    For example, if we are comparing A1 with A2, A3, A4 and A5 then, the polar coordinate
    system is divided in four sectors, in which, from [0, 90] degrees we represent the
    probability that A1 is better than A2 and so on.

    The probability that an algorithm is better than another is represented by drawing
    one point per posterior sample in the corresponding sector of the scatter plot.
    For example, if algorithm A1 is better than algorithm A2 with probability p
    then, we generate a point in polar coordinates (r, p) with r ∼ unif(0, 90).
    

    Parameters:
    -----------

    probs : list of ndarray
      A 3-dimensional matrix returned by calculate_better_than_probs().

    empirical: list of float
      A list of empirical probabilities for each algorithm being compared.

    axs : list of matplotlib axes
      A list containing matplotlib axes in polar coordinates, one per algorithm being considered.
      In case some axis is None, the method will skip that particular plot.
    """

    num_algorithms = len(algorithm_names)

    for i, ax in enumerate(axs):
      if ax != None:
        minor = []
        major = []
        labels =  []

        for j, idx in enumerate([idx for idx in range(num_algorithms) if idx != i]):
            start = 2 * np.pi * j / (num_algorithms - 1)
            end = start + 2 * np.pi / (num_algorithms - 1)

            major.append(start)
            major.append(end)
            minor.append((end + start) / 2)
            labels.append(algorithm_names[idx])
              
        ax.set_xticks(major)
        ax.set_xticks(minor, minor=True)
        ax.set_xticklabels(labels, minor=True, fontsize=18)
        ax.set_xticklabels([], minor=False)

        ax.text(0, 0, algorithm_names[i], ha='center',va='center', color='blue', fontsize=24, bbox=dict(facecolor=(1, 1, 1, 0.5), edgecolor='none', boxstyle='circle'))
        ax.grid(which='minor', alpha=0.4, linewidth=0.1, color='black')
        ax.grid(which='major', alpha=1, linewidth=1)
        ax.grid(True)

        for j, idx in enumerate([idx for idx in range(num_algorithms) if idx != i]):
            p = probs[:, i, idx]
            start = 2 * np.pi * j / (num_algorithms - 1)
            end = start + 2 * np.pi / (num_algorithms - 1)

            theta = np.random.uniform(start, end, p.shape)
            ax.scatter(theta, p, color='gray', alpha=0.1)

  def plot_top_k_probs(self, algorithm_names, probs, empirical, axs):
    """ Plot the probability that each algorithm is in the top-k ranking.

    This function generates several violin plot, oneecond recommendation goes in the
direction of conducting a sensitivity analysis to verify if the general overview and conclusions are
affected when we vary the number of samples after some point. A third alternative is to conduct
a cross-validation analysis as long as we have sufficient experimental data to do s per possible top-k ranking.
    In the horizontal axis of each plot we represent the different algorithms being compared,
    If given, the function plots a black dot that represents the empirical probability.

    Parameters:
    -----------

    model_names : list of string
      A list of strings representing the different algorithm names.

    probs : list of ndarray
      A 3-dimensional matrix returned by calculate_top_k_probs().

    empirical: list of float
      A list of empirical probabilities for each algorithm being compared.

    axs : list of matplotlib axes
      A list containing matplotlib axes, one per algorithm being considered. In case some
      axis is None, the method will skip that particular plot.
    """

    num_algorithms = len(algorithm_names)

    for i, ax in enumerate(axs):
      if ax != None:
        idxs = list(range(num_algorithms))
        names = algorithm_names
        x_values = range(1, len(idxs) + 1)
        empirical_values = [empirical[i, vs] for vs in idxs]
        
        ax.violinplot(probs[:, idxs, i], showmeans=True)
        ax.scatter(x_values, empirical_values, marker='o', color='black', s=30, zorder=3)
        self.set_axis_style(ax, names)

        ax.set_title("Top: " + str(i + 1))

