import numpy as np
import itertools

def tabuleTopOne(probs):
  num_models = len(probs)
  mixed = []
    
  for model in range(num_models):
      mean = np.mean(probs[model], axis=0)
      std = np.std(probs[model], axis=0)
      
      row_mixed = []
      for m, s in zip(mean, std):
        row_mixed.append("{:,.2E}".format(m) + " (" + "{:,.2E}".format(s) + ")")
      mixed.append(row_mixed)

  return mixed

def tabuleBetterThan(probs):
  num_models = len(probs)
  mixed = []
    
  for model in range(num_models):
      mean = np.mean(probs[model], axis=0)
      std = np.std(probs[model], axis=0)
      
      for row_mean, row_std in zip(mean, std):
          row_mixed = []
          
          for m, s in zip(row_mean, row_std):
              row_mixed.append("{:,.2E}".format(m) + " (" + "{:,.2E}".format(s) + ")")
              
          mixed.append(row_mixed)

  return mixed

def tabuleTopK(probs):
  num_models = len(probs)
  mixed = []
    
  for model in range(num_models):
      mean = np.mean(probs[model], axis=0)[:,1:-1]
      std = np.std(probs[model], axis=0)[:,1:-1]
      
      for row_mean, row_std in zip(mean, std):
          row_mixed = []
          
          for m, s in zip(row_mean, row_std):
              row_mixed.append("{:,.2E}".format(m) + " (" + "{:,.2E}".format(s) + ")")
              
          mixed.append(row_mixed)

  return mixed