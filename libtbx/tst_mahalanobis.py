from __future__ import division
from io import StringIO

diamonds_short = '''"carat","cut","color","clarity","depth","table","price","x","y","z"
0.23,"Ideal","E","SI2",61.5,55,326,3.95,3.98,2.43
0.21,"Premium","E","SI1",59.8,61,326,3.89,3.84,2.31
0.23,"Good","E","VS1",56.9,65,327,4.05,4.07,2.31
0.29,"Premium","I","VS2",62.4,58,334,4.2,4.23,2.63
0.31,"Good","J","SI2",63.3,58,335,4.34,4.35,2.75
0.24,"Very Good","J","VVS2",62.8,57,336,3.94,3.96,2.48
0.24,"Very Good","I","VVS1",62.3,57,336,3.95,3.98,2.47
0.26,"Very Good","H","SI1",61.9,55,337,4.07,4.11,2.53
0.22,"Fair","E","VS2",65.1,61,337,3.87,3.78,2.49
0.23,"Very Good","H","VS1",59.4,61,338,4,4.05,2.39
0.3,"Good","J","SI1",64,55,339,4.25,4.28,2.73
0.23,"Ideal","J","VS1",62.8,56,340,3.93,3.9,2.46
0.22,"Premium","F","SI1",60.4,61,342,3.88,3.84,2.33
0.31,"Ideal","J","SI2",62.2,54,344,4.35,4.37,2.71
0.2,"Premium","E","SI2",60.2,62,345,3.79,3.75,2.27
0.32,"Premium","E","I1",60.9,58,345,4.38,4.42,2.68
0.3,"Ideal","I","SI2",62,54,348,4.31,4.34,2.68
0.3,"Good","J","SI1",63.4,54,351,4.23,4.29,2.7
0.3,"Good","J","SI1",63.8,56,351,4.23,4.26,2.71
'''

zz=[(0.23, 61.5, 326),
    (0.23, 56.9, 329),
    (0.23, 56.9, 349),
   ]

def mahalanobis_using_pandas():
  import numpy as np
  import pandas as pd
  import scipy as stats
  from scipy.stats import chi2

  # calculateMahalanobis Function to calculate
  # the Mahalanobis distance
  def calculateMahalanobis(y=None, data=None, cov=None):

      y_mu = y - np.mean(data)
      print(data)
      print(y)
      print(y_mu)
      if not cov:
        cov = np.cov(data.values.T)
        print(cov)
      inv_covmat = np.linalg.inv(cov)
      left = np.dot(y_mu, inv_covmat)
      mahal = np.dot(left, y_mu.T)
      return mahal.diagonal()

  # data
  data = { 'Price': [100000, 800000, 650000, 700000,
                     860000, 730000, 400000, 870000,
                     780000, 400000],
           'Distance': [16000, 60000, 300000, 10000,
                        252000, 350000, 260000, 510000,
                        2000, 5000],
           'Emission': [300, 400, 1230, 300, 400, 104,
                        632, 221, 142, 267],
           'Performance': [60, 88, 90, 87, 83, 81, 72,
                           91, 90, 93],
           'Mileage': [76, 89, 89, 57, 79, 84, 78, 99,
                       97, 99]
             }

  # Creating dataset
  df = pd.DataFrame(data,columns=['Price', 'Distance',
                                  'Emission','Performance',
                                  'Mileage'])

  # Creating a new column in the dataframe that holds
  # the Mahalanobis distance for each row
  data = { 'Price': [100000],
           'Distance': [60000],
           'Emission': [104,
                        ],
           'Performance': [ 87],
           'Mileage': [99]
             }
  y=pd.DataFrame(data,columns=['Price', 'Distance',
                                  'Emission','Performance',
                                  'Mileage'])
  rc = calculateMahalanobis(y=y, data=df[[
    'Price', 'Distance', 'Emission','Performance', 'Mileage']])
  print(rc)

  # calculate p-value for each mahalanobis distance
  rc = 1 - chi2.cdf(rc, 4)
  print(rc)

  # display first five rows of dataframe
  print(df)
  assert 0

def another_pandas():
  import numpy as np

  def mahalanobis(x, y, cov=None):
      x_mean = np.mean(x,0)
      Covariance = np.cov(np.transpose(y))
      inv_covmat = np.linalg.inv(Covariance)
      x_minus_mn = x - x_mean
      D_square = np.dot(np.dot(x_minus_mn, inv_covmat), np.transpose(x_minus_mn))
      return D_square.diagonal()

  import pandas as pd

  # filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/diamonds.csv'
  filepath='diamonds_short.csv'
  f=open(filepath, 'w')
  f.write(diamonds_short)
  del f
  df = pd.read_csv(filepath).iloc[:, [0,4,6]]
  df.head()

  #"carat","cut","color","clarity","depth","table","price","x","y","z"
  data = { 'carat': [],
           'depth': [],
           'price': []
        }
  for i in range(len(zz)):
    print(i,zz[i])
    data['carat'].append(zz[i][0])
    data['depth'].append(zz[i][1])
    data['price'].append(zz[i][2])
  print(data)

  X=pd.DataFrame(data,columns=['carat', 'depth', 'price'])
  print(X)

  X = np.asarray(X[['carat', 'depth', 'price']].values)
  Y = np.asarray(df[['carat', 'depth', 'price']].values)
  print(X)
  print(Y)

  rc=mahalanobis(X, Y)
  print('another_pandas',rc)

def mahalanobis_using_numpy(x, data=None, cov=None, verbose=False):
  import numpy as np
  def mahalanobis(x, data, cov=None):
    x_mean = np.mean(x, 0)
    if verbose: print('x_mean',x_mean)
    if cov:
      Covariance=np.asarray(cov)
    else:
      Covariance = np.cov(np.transpose(data))
    if verbose:
      print('Covariance')
      print(Covariance)
    inv_covmat = np.linalg.inv(Covariance)
    if verbose:
      print('inv_covmat')
      print(inv_covmat)
    x_minus_mn = x - x_mean
    if verbose:
      print('x_minus_mn')
      print(x_minus_mn)
    D_square = np.dot(np.dot(x_minus_mn, inv_covmat), np.transpose(x_minus_mn))
    if verbose:
      print('D_square')
      print(D_square)
      print(D_square.diagonal())
    return D_square.diagonal()

  if verbose:
    print(x)
    print(data)
  rc = mahalanobis(x, data=data, cov=cov)
  return rc

def mahalanobis_using_numpy_and_scipy(x=None, data=None, mu=None, cov=None):
  """
  Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each
           observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be
           computed from data.
  """
  import numpy as np
  import scipy as sp

  x=np.array(x)
  print(x)
  data=np.array(data)
  print(data)
  x_mu = x - np.mean(data)
  if not cov:
      cov = np.cov(data.values.T)
  inv_covmat = np.linalg.inv(cov)
  left = np.dot(y_mu, inv_covmat)
  mahal = np.dot(left, y_mu.T)
  return mahal.diagonal()

  if mu:
    x_minus_mu = x - mu
  else:
    print(x)
    print(np.mean(data))
    x_minus_mu = x - np.mean(data)
    print('x_minus_mu',x_minus_mu)
    assert 0
  # x_minus_mu=np.array(x)
  print('x_minus_mu',x_minus_mu)
  if not cov: cov = np.cov(data)
  print(cov)
  inv_covmat = sp.linalg.inv(cov)
  print(inv_covmat)
  left_term = np.dot(x_minus_mu, inv_covmat)
  mahal = np.dot(left_term, x_minus_mu.T)
  return mahal.diagonal()

def main():
  from libtbx import math_utils
  import csv
  data=[]
  f=StringIO(diamonds_short)
  spamreader = csv.reader(f, delimiter=',', quotechar='|')
  for i, row in enumerate(spamreader):
    if not i: continue
    print(i,', '.join(row))
    data.append([float(row[0]), float(row[4]), float(row[6])])

  use_numpy=0
  use_pandas=0
  if use_pandas:
    another_pandas()

  if use_numpy:
    # another_pandas()
    # mahalanobis_using_pandas()
    #np.asarray
    rc1 = mahalanobis_using_numpy(zz, data, verbose=1)
    # rc = mahalanobis_using_numpy_and_scipy(zz, data)
    print('rc1',rc1)
    rc2 = mahalanobis_using_numpy(zz, cov=[
      [1.55789474e-03, 3.08947368e-02, 1.42105263e-01],
      [3.08947368e-02, 3.50038781e+00, 5.25207756e+00],
      [1.42105263e-01, 5.25207756e+00, 5.39556787e+01]], verbose=1)
    print('rc2',rc2)

  rc=math_utils.mahalanobis(zz, data)
  print(rc)
  rc=math_utils.mahalanobis_p_values(zz, data, verbose=True)
  print(rc)
  rc=math_utils.mahalanobis_p_values_outlier_indices(zz, data)
  assert rc==[2]
  for i, p in enumerate(math_utils.mahalanobis_p_values(zz, data)):
    outlier=''
    if i in rc:
      outlier='OUTLIER'
    print('  %s %0.4f %s' % (i,p,outlier))
  cov = math_utils.covariance_using_sklearn(data, verbose=True)
  rc=math_utils.mahalanobis(zz, cov=cov)
  print(rc)
  if use_numpy:
    print(rc1)
    print(rc2)
  rc=math_utils.mahalanobis_p_values(zz, cov=cov, verbose=True)
  print(rc)
  rc=math_utils.mahalanobis_p_values_outlier_indices(zz, cov=cov)
  assert rc==[2]
  rc=math_utils.mahalanobis_p_values_outlier_indices(zz[:1], cov=cov)
  print(rc)

  # values = cov.covariance_.tolist()
  # cov = math_utils.covariance_using_sklearn(values=values, verbose=True)
  # assert 0
  from libtbx import easy_pickle
  easy_pickle.dump('tst_mahal.pickle', cov)
  cov=easy_pickle.load('tst_mahal.pickle')
  # import pickle
  # pf='tst_mahal.pickle'
  # f=open(pf, 'w')
  # pickle.dump(cov, f)
  # del f
  # f=open(pf, 'r')
  # cov=pickle.load(f)
  # del f
  rc=math_utils.mahalanobis_p_values_outlier_indices(zz, cov=cov)
  print(rc)
  assert rc==[2]

if __name__ == '__main__':
  main()
