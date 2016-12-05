import numpy as np
import cPickle
import heapq
import os
from IPython.core.debugger import Tracer
import scipy.io as scio
import time
from scipy.interpolate import interp1d

def tic_toc_print(interval, string):
  global tic_toc_print_time_old
  if 'tic_toc_print_time_old' not in globals():
    tic_toc_print_time_old = time.time()
    print string
  else:
    new_time = time.time()
    if new_time - tic_toc_print_time_old > interval:
      tic_toc_print_time_old = new_time;
      print string

def mkdir_if_missing(output_dir):
  """
  def mkdir_if_missing(output_dir)
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_variables(pickle_file_name, var, info, overwrite = False):
  """
    def save_variables(pickle_file_name, var, info, overwrite = False)
  """
  if os.path.exists(pickle_file_name) and overwrite == False:
    raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
  # Construct the dictionary
  assert(type(var) == list); assert(type(info) == list);
  d = {}
  for i in xrange(len(var)):
    d[info[i]] = var[i]
  with open(pickle_file_name, 'wb') as f:
    cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)

def load_variables(pickle_file_name):
    if os.path.exists(pickle_file_name):
        with open(pickle_file_name, 'rb') as f:
            d = cPickle.load(f)
        return d
    else:
        raise Exception('{:s} does not exists.'.format(pickle_file_name))

def generate_map(thresh, prec):
    ind = np.argsort(thresh);
    thresh = thresh[ind];
    prec = prec[ind];
    for i in xrange(1, len(prec)):
        prec[i] = max(prec[i], prec[i-1]);

    indexes = np.unique(thresh, return_index=True)[1]
    indexes = np.sort(indexes);
    thresh = thresh[indexes]
    prec = prec[indexes]
    thresh = np.vstack((0, thresh[:, np.newaxis], 1));
    prec = np.vstack((0, prec[:, np.newaxis], 1));
    f = interp1d(thresh[:,0], prec[:,0])
    return f
