import numpy as np
import pandas as pd
import itertools

def fix_index(df):
  fixed_index = []
  problems = []

  for problem, rep in df.index:
    if type(problem) == str:
      problems.append(problem)
      prev = problem
    fixed_index.append((prev, rep))

  fixed_index = pd.MultiIndex.from_tuples(fixed_index)
  return fixed_index, problems

def ranks_from_score(score):
  # Set of linear extensions of the original ranking. The linear extensions are
  # obtained when ties are resolved in all possible ways.
  permus = []
  weights = []

  n = len(score)
  rank = np.argsort(score)

  # List that contains several lists that represent element's index that are 
  # repeated in the original score.
  #
  # For example, if there is a single list with two elements, e.g. [[3, 6]] it
  # means that `score[3] == score[6].
  ties_set = []
  excluded = []
  
  # Loop through all elements in score.
  for i in range(n):
    if i not in excluded:
      repeated = [i]

      # Check if there are any ties in the rest of the score list.
      for j in range(i + 1, n):

        # If there is a tie, then add the entry to the repeated list.
        if score[i] == score[j]:
          repeated.append(j)

      excluded += repeated
      
      # If there is any tie, then, add it to the tie set.
      if len(repeated) > 1:
        ties_set.append(repeated)

  # List that contains several lists that represent the elements index that are 
  # repeated in the original score and all their permutations. 
  #
  # For example, if there is a single list with two elements, it means that there 
  # is a single repeated element that appears three times in the original score.
  #
  # If there are two lists with three elements each, it means that there are two
  # repeated elements that appear three times each in the original score.
  #
  # The values within each list inside this list represent the rankings of such
  # repeated entries.
  extensions = []
  for i, repeated in enumerate(ties_set):
    extensions.append(list(itertools.permutations(repeated)))
  extensions = list(itertools.product(*extensions))

  # Loop through all possible linear extensions.
  for extension in extensions:
    # Start to modify the original ranking to create the linear extension.
    permu = rank

    # Swap the rankings of the repeated / tie scores iteratively.
    for section in extension:
      # Size of the section to be replaced.
      sec_size = len(section)
      
      # Determine the starting point in ranking in which we replace
      # the section.
      for start, value in enumerate(permu):
        if value in section:
          break
      
      # Modify the original ranking iteratively.
      permu = np.concatenate((permu[:start], section, permu[start + sec_size:]))

    # Add the linear extension to the permutation set.
    permus.append(permu + 1)
    weights.append(1.0 / len(extensions))

  return permus, weights

def load_permus_from_file(prefix, algorithms, num_instances=10, num_reps=20):
  permus = []
  scores = []
  problems = []
  weights = [] 
  dfs = [pd.read_csv(prefix + '-' + algorithm + '.csv', header=[0], 
                     index_col=[0,1]) for algorithm in algorithms]

  for df in dfs:
    index, problems = fix_index(df)
    df.index = index
    df = df.astype(int)

  for i, problem in enumerate(problems):
      for instance in range(num_instances):
        for rep in range(num_reps):
          # Score of each algorithm.
          score = []

          for df in dfs:
            # Locate the score for each algorithm per problem / instance / rep.
            score.append(df.loc[(problem, str(rep + 1)), str(instance)])
            # Obtain the rankings, including linear extensions.
          
          p, w = ranks_from_score(score)
          scores.append(score)

          permus += p
          weights += w

  return np.array(permus), np.array(weights), np.array(scores)

def sample_permus(permus, weights, num_samples):
  n = len(weights)
  sample_permus = []
  sample_weights = []

  while True:
    idx = np.random.randint(0, n)

    permu = permus[idx]
    w = weights[idx]

    if np.random.random() < w:
      sample_permus.append(permu)
      sample_weights.append(w)

      if len(sample_weights) == num_samples:
        return np.array(sample_permus), np.array(sample_weights)