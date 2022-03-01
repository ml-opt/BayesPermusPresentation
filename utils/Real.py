import numpy as np
import pandas as pd
import itertools
import random

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

  while True:

    extension = []

    for tie in ties_set:
      random.shuffle(tie)
      extension.append(tie)
    
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

    yield permu + 1

def load_permus_from_CEB(prefix, algorithms, num_instances=10, num_reps=20):
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
          
          generator = ranks_from_score(score)
          
          permus.append(generator)
          scores.append(score)

  return permus, np.array(scores)

def load_permus_from_CEC(prefix, problems, dimensions, errors, algorithms, repetitions):
  permus = []
  scores = []

  for problem in problems:
    for dimension in dimensions:
      results = [pd.read_csv(prefix + '/' + algorithm + '/' + str(algorithm) + '_' + 
                             str(problem) + '_' + str(dimension) + '.txt',
                             header=None, sep=",") for algorithm in algorithms]

      for error in errors:
        for repetition in repetitions:
          score = []

          for result in results:
            score.append(result.iloc[error, repetition])

          generator = ranks_from_score(score)
          scores.append(score)
          permus.append(generator)
  
  return permus, np.array(scores)

def sample_permus(permus, num_samples):
  n = len(permus)
  sample_permus = []

  for i in range(num_samples):
    idx = np.random.randint(0, n)
    permu = next(permus[idx])
    sample_permus.append(permu)

  return np.array(sample_permus)