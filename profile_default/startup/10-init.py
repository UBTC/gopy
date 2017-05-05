# !/usr/bin/env python
# -*- coding:utf-8 -*-
# see `%pylab?` from ipython for help (ipython --pylab)

# try to work with ipython3 (&ipython)
# http://python-future.org/compatible_idioms.html
# To make Py2 code safer by preventing implicit relative imports
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import future        # pip install future (I used pip3)
import builtins    # remove when running on cluster!!!
import past    # remove when running on cluster!!!
# import six           # pip install six (pip3???)

from past.builtins import map    # remove when running on clustervvv
from past.builtins import range  #                               !!!
from past.builtins import execfile
from future.utils import viewitems # iterating through dict values/items
from future.utils import viewvalues
from future.utils import iteritems
from future.utils import itervalues
from future.utils import listitems
from future.utils import listvalues #                            !!!
from functools import reduce
from builtins import input       # remove when running on cluster^^^

import re
import os
import sys
import mdp    # remove when running on cluster!!!
import numpy
import scipy
import string
import pandas    # remove when running on cluster!!!
import networkx
import itertools
import matplotlib
# import plotly.plotly
# import plotly.graph_objs
import scipy.optimize
import statsmodels.api    # remove when running on cluster!!!
import statsmodels.formula.api    # remove when running on cluster!!!
np = numpy
sp = scipy
pd = pandas
nx = networkx
mpl = matplotlib
opt = scipy.optimize
# py = plotly.plotly
# go = plotly.graph_objs
sm = statsmodels.api    # remove when running on cluster!!!
smf = statsmodels.formula.api    # remove when running on cluster!!!

try: # Python2:
    from itertools import imap
except ImportError: # Python3:
    imap=map

#mpl.use("pdf")     # uncomment when running on cluster!!!

from functools import reduce
from scipy import stats, linalg
from matplotlib import pylab, mlab, pyplot
from numpy.random import rand, randn
from numpy.polynomial import polynomial
from pandas import DataFrame, Series    # remove when running on cluster!!!
from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs
from datetime import *
plt = pyplot
plab = pylab

from pylab import *
from numpy import *
from scipy import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# import importlib
# importlib.import_module('mpl_toolkits.mplot3d').Axes3D
from scipy import interpolate

true = True
false = False


def datetime_str(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


print()
print(datetime_str())


# ... ... ....

def smooth(x):
    # 1 2 1 smooth on 1d array
    if array(x).ndim != 1:
        print('smooth can only be applied on 1d array')
        return -1
    y=zeros(shape(x))
    for i in range(1,len(x)-1):
        y[i]=(x[i-1]+2*x[i]+x[i+1])/4.0
    y[0],y[-1]=x[0],x[-1]
    return y

def sets_divergence(A,B): # A and B are both sets
    """
        The Jaccard distance measures dissimilarity between sample sets.
        It is obtained by dividing the difference of the sizes of
           the union and the intersection of two sets, by the size of the union.
        https://en.wikipedia.org/wiki/Jaccard_index
    """
    if isinstance(A,list) or isinstance(A,numpy.ndarray): A=set(A)
    if isinstance(B,list) or isinstance(B,numpy.ndarray): B=set(B)
    return 1.0*(len(A.union(B))-len(A.intersection(B)))/len(A.union(B))


def Jaccard_distance(A,B): return sets_divergence(A,B)


def PCA(data, nComp=3, svas=''):
    " input dim is [x, y], where x is timebin number, and y is PN number "
    # return mdp.pca(x) # see also
    # http://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
    if isinstance(data, list): data = array(data)
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=0)
    if svas!='': np.savetxt(svas, R)
    eVals, eVecs = linalg.eigh(R)
    idx = np.argsort(eVals)[::-1]
    eVecs = eVecs[:, idx]
    eVals = eVals[idx]
    eVecs = eVecs[:, :nComp]
    if if_ret_all:
        return np.dot(data, eVecs), eVals, eVecs
    return np.dot(data, eVecs)


def PCAtp(d, n=3):
    return PCA(d.T, n).T


def CAby(data, mat, nComp=3):
    " input dim is [x, y], where x is timebin number, and y is PN number "
    if isinstance(data, list): data = array(data)
    if isinstance(mat,  str):
        R = loadtxt(mat)
    elif isinstance(mat, list):
        R = array(mat)
    else: R = mat
    eVals, eVecs = linalg.eigh(R)
    idx = np.argsort(eVals)[::-1]
    eVecs = eVecs[:, idx]
    eVals = eVals[idx]
    eVecs = eVecs[:, :nComp]
    return np.dot(eVecs.T, data.T).T


def CAbytp(d, m, n=3):
    return PCA(d.T, m, n).T


def myPSD(data, Fs, NFFT):
    """
      this function return the power and corresponding freq of data
      ...
      the original data is transferred via `data`, which is usually an 1d list.
      Fs and NFFt are used to compute power spectral density by Welches method.
      data is divided into NFFT length segments. Fs is the sampling frequency.
      ... ... ...
      # An example calling:
      PNnum = 830
      nfftExp = 10 # !!!
      a = loadtxt("doc_V_PN_c0_s0_t0.txt")[1:, 1:]
      b = sum(a, 1)
      p, f = myPSD(b/PNnum, 1000, 2**nfftExp)
      figure()
      plot(f[:90], p[:90])
      xlabel('freq (Hz)')
      ylabel('power (dB/Hz?)')
      savefig("psd.eps")
      show()
    """
    p,f = psd(data-mean(data), Fs, NFFT)
    clf()
    return p,f


def bandpower(data, Fs, NFFT, winLen, winStepLen, lowerLimit, upperLimit):
    """
      this function return the bandpower in [lowerLimit, upperLimit]
      the original data is transferred via `data`, and is cutted to windows.
      The windows are defined by winLen and winSteplen
      ...
      Fs and NFFt are used to compute power spectral density by Welches method.
      data is divided into NFFT length segments. Fs is the sampling frequency.
      ...
      The returned array gives the bandpower at each window.
      ... ... ...
      # An example calling:
      PNnum = 830
      nfftExp = 10 # !!!
      wLen = 200
      wStepLen = 50
      a = loadtxt("doc_V_PN_c0_s0_t0.txt")[1:, 1:]
      b = sum(a, 1)
      all_win_val = bandpower(b/PNnum, 1000, 2**nfftExp, wLen, wStepLen, 15, 25)
      #show()
      # ...
      x_ticks = [0,20,40,60,80,100]
      x_label = [0, 1, 2, 3, 4,  5]
      figure()
      plot(all_win_val)
      # axhline(y=-0.05, xmin=250, xmax=750, linewidth=4, color='b')
      plot([28, 70], [0, 0], linewidth=8, color='black')
      plot([20, 90], [0, 0], linewidth=4, color='black')
      xticks(x_ticks, x_label)
      xlabel('time (S)')
      ylabel('bandpower ~20Hz')
      savefig("bandpower_20Hz.eps")
      show()
    """
    all_win_val = []
    for wi_begin in range(0, len(data)-winLen+winStepLen, winStepLen): # window, the i step
        wi_end = wi_begin + winLen
        p, f = psd(data[wi_begin:wi_end] - mean(data[wi_begin:wi_end]), Fs, NFFT)
        ttt = 0
        for jjj, xxx in enumerate(f):
            if xxx > 15 and xxx <= 25:
                ttt += p[jjj]
        all_win_val.append(ttt)
    clf()
    return array(all_win_val)


# plot functions with Matplotlib
# fplot('x**3+2*x-4')
# fplot('y=x**3+2*x-4', [-10, 10, 100])
# https://stackoverflow.com/questions/14000595/graphing-an-equation-with-matplotlib
def fplot(formula, xmms=[-5,5]): # mms is min, max [and steps]
    if len(xmms)==3:
        x = np.linspace(xmms[0], xmms[1], xmms[2])
    else:
        x = np.linspace(xmms[0], xmms[1], 1000)
    # ...
    t = formula.find('=')
    if t>0: # it looks like 'y=x'
        y = eval(formula[t+1:])
    else: # it looks like 'x'
        y = eval(formula)
    # ...
    return plt.plot(x, y)
    # plt.show()


def jittering(lls, randScale=0.01, sampleNum=3):  # a simple jittering function
    # http://matplotlib.1069221.n5.nabble.com/jitter-in-matplotlib-td12573.html
    return stats.norm.rvs(loc=lls, scale=randScale, size=(sampleNum, len(lls)))

"""
xs,ys = np.random.random((2,5))
plt.scatter(xs, ys, c='b')
# create jittered data for x and y coords
xs_jit = jittering(xs)
ys_jit = jittering(ys)
plt.scatter(xs_jit, ys_jit, c='r')
plt.show()
"""


def enum(x): # a shoter enumerate
    return enumerate(x)


def rlen(x):
    return range(len(x))
# b=ones(10)
# for i in rlen(b): print(i, b[i])
# 0 1.0
# 1 1.0
# 2 1.0
#  ...
# 9 1.0


"""
muloop([12])
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

muloop([12.1])
TypeError: 'float' object cannot be interpreted as an integer

muloop([12,1])
<itertools.product at 0x...>

muloop([12,-1])
<itertools.product at 0x...>

for i,j in muloop([12,-2]):
    print(i,j)
[empty]

for i,j in muloop([12,2]):
    print(i,j)
[multi-loops]
"""

def muloop(x):
    if not (isinstance(x,list) or isinstance(x,numpy.ndarray) or isinstance(x,tuple)):
        print("\nERROR: muloop must have a list or array (with 1-10 items) as parameter!")
        return []
    if len(x)==0:
        print("\nERROR: muloop should have a list or array with 1-10 items as parameter, but received", len(x))
        return []
    elif len(x)==1:
        return range(x[0])
    elif len(x)==2:
        return itertools.product(range(x[0]), range(x[1]))
    elif len(x)==3:
        return itertools.product(range(x[0]), range(x[1]), range(x[2]))
    elif len(x)==4:
        return itertools.product(range(x[0]), range(x[1]), range(x[2]), range(x[3]))
    elif len(x)==5:
        return itertools.product(range(x[0]), range(x[1]), range(x[2]), range(x[3]), range(x[4]))
    elif len(x)==6:
        return itertools.product(range(x[0]), range(x[1]), range(x[2]), range(x[3]), range(x[4]), range(x[5]))
    elif len(x)==7:
        return itertools.product(range(x[0]), range(x[1]), range(x[2]), range(x[3]), range(x[4]), range(x[5]), range(x[6]))
    elif len(x)==8:
        return itertools.product(range(x[0]), range(x[1]), range(x[2]), range(x[3]), range(x[4]), range(x[5]), range(x[6]), range(x[7]))
    elif len(x)==9:
        return itertools.product(range(x[0]), range(x[1]), range(x[2]), range(x[3]), range(x[4]), range(x[5]), range(x[6]), range(x[7]), range(x[8]))
    elif len(x)==10:
        return itertools.product(range(x[0]), range(x[1]), range(x[2]), range(x[3]), range(x[4]), range(x[5]), range(x[6]), range(x[7]), range(x[8]), range(x[9]))
    else:
        print("\nERROR: muloop should have a list or array with 1-10 items as parameter, but received", len(x))
        return []


# https://github.com/sciy/temFlow/blob/master/list_process.py
def outter_flatten(lst):
    """only flat the outter level"""
    new_lst = []
    for x in lst:
        if isinstance(x, list):
            for y in x: new_lst.append(y)
        else:
            new_lst.append(x)
    return new_lst

# In [-]: a
# Out[-]: [[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6]], [[7, 8, 9], [7, 8, 9]]]
# In [-]: shape(a)
# Out[-]: (3, 2, 3)
# In [-]: outter_flatten(a)
# Out[-]: [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [7, 8, 9], [7, 8, 9]]
# In [-]: shape(outter_flatten(a))
# Out[-]: (6, 3)
# #-----------
# In [-]: len(shape(a))
# Out[-]: 3
# In [-]: a=outter_flatten(a)
# In [-]: len(shape(a))
# Out[-]: 2
# In [-]: a=outter_flatten(a)
# In [-]: len(shape(a))
# Out[-]: 1
# In [-]: a
# Out[-]: [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9]


def equally_divide(lst, segment_len):
    if len(lst)%segment_len != 0: lst = lst[:int(floor(len(lst)/segment_len*segment_len))]
    return [lst[i:i+segment_len] for i in range(0, len(lst), segment_len)]

# In [-]: a
# Out[-]: [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9]
# In [-]: a=equally_divide(a,3)
# Out[-]: [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [7, 8, 9], [7, 8, 9]]
# In [-]: a=equally_divide(a,2)
# Out[-]: [[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [4, 5, 6]], [[7, 8, 9], [7, 8, 9]]]
# In [-]: shape(a)
# Out[-]: (3, 2, 3)


def avg(x):
    # avg([2,3,4])
    # == 3.0
    return np.mean(x)


def minN(a, n):
    if not isinstance(a, list) or not isinstance(a, ndarray): return False
    if n>len(a): n=len(a)
    b = a[:]
    for i in range(len(a)): b[i] = (b[i], i)
    b.sort(key = lambda x: x[0], reverse = False)
    return array([b[i][0] for i in range(n)]), array(map(int, [b[i][1] for i in range(n)]))


def maxN(a, n):
    if not isinstance(a, list) or not isinstance(a, ndarray): return False
    if n>len(a): n=len(a)
    b = a[:]
    for i in range(len(a)): b[i] = (b[i], i)
    b.sort(key = lambda x: x[0], reverse = True)
    return array([b[i][0] for i in range(n)]), array(map(int, [b[i][1] for i in range(n)]))

#In [-]: a=[13,4,23,9,111]
#In [-]: maxN(a, 3)
#Out[-]: ([111, 23, 13], [4, 2, 0])
#
#In [-]: minN(a, 3)
#Out[-]: ([4, 9, 13], [1, 3, 0])
#
#In [-]: minN(a, 33)
#Out[-]: ([4, 9, 13, 23, 111], [1, 3, 0, 2, 4])


# handy 2d fitting function
# http://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
# do the fitting:
def polyfit2d(x, y, z, deg):
    # deg : x and y maximum degrees: [x_deg, y_deg].
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1, vander.shape[-1]))
    z = z.reshape((vander.shape[0], ))
    c = np.linalg.lstsq(vander, z)[0]
    return c.reshape(deg+1)


# get the fitting z results at given (x,y) points
#     used for doing plots:
def polyval2d(x, y, m):
    ij = itertools.product(range(shape(m)[0]), range(shape(m)[1]))
    z = np.zeros_like(x)
    for a, (i,j) in zip(flatten(m), ij):  z = z + a * x**i * y**j
    return z

"""
a=array([ [i, j, i**2+(100-j)**2]  for i in range(100)  for j in range(100) ])

m = polyfit2d(a[:,0], a[:,1], a[:,2], [3,3]) # fits it!

m is :
array([[  1.00000033e+04,  -2.00000024e+02,   9.99998625e-01,  1.18931529e-08],
       [ -4.10125405e-05,  -1.30057323e-06,   5.62423528e-08, -3.80531578e-10],
       [  9.99998620e-01,   5.63967447e-08,  -6.96189863e-10,  3.22986082e-12],
       [  1.18394489e-08,  -3.79559266e-10,   3.20454774e-12, -2.84217094e-14]])


The above matrix m gives coefficients of (x, y)
(0, 0) (0, 1) (0, 2) (0, 3)
(1, 0) (1, 1) (1, 2) (1, 3)
(2, 0) (2, 1) (2, 2) (2, 3)
(3, 0) (3, 1) (3, 2) (3, 3)


tmp = polyval2d(a[:,0], a[:,1], m) # compute the fitting vals at each point

plot(abs(tmp-a[:,2])) # checherrors
"""


# Default styles for Matplotlib, makes it a better MPL,
#     stolen from charnley, revised to make it works with python 3:
#     github.com/charnley/matplotlib-header/blob/master/mpl_header.py


### FIGURE (This section was at the bottom. Now only this section is used.)
# use the ggplot style for matplotlib 1.4 +
plt.style.use('ggplot')

# See http://matplotlib.sourceforge.net/api/figure_api.html#matplotlib.figure.Figure
plt.rc('figure', figsize=(12, 9), dpi=300)    # figure size in inches
# TODO Set for 1 column size (with crop please)


### Fonts
# # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# # plt.rc('font',**{'family':'serif','serif':['Palatino']})
# plt.rc('text', usetex=True)     # Use Latex formatting
# plt.rc('text.latex', preamble='\\usepackage{helvet}') # Use sans-serif font
# plt.rc('font',       size=15)         # Fontsize
# plt.rc('xtick', labelsize=15)   # Fontsize for x-ticks
# plt.rc('ytick', labelsize=15)   # Fontsize for y-ticks
# plt.rc('legend', fontsize=15)


## Colors
# Default colors are
# {'c': (0.0, 0.75, 0.75),
#  'b': (0.0, 0.0, 1.0),
#  'w': (1.0, 1.0, 1.0),
#  'g': (0.0, 0.5, 0.0),
#  'y': (0.75, 0.75, 0),
#  'k': (0.0, 0.0, 0.0),
#  'r': (1.0, 0.0, 0.0),
#  'm': (0.75, 0, 0.75)}

def hex2color(s):
    """
    Take a hex string *s* and return the corresponding rgb 3-tuple
    Example: #efefef -> (0.93725, 0.93725, 0.93725)    """
    hexColorPattern = re.compile("\\A#[a-fA-F0-9]{6}\\Z")
    if not isinstance(s, str):
        raise TypeError('hex2color requires a string argument')
    if hexColorPattern.match(s) is None:
        raise ValueError('invalid hex color string "%s"' % s)
    return tuple([int(n, 16)/255.0 for n in (s[1:3], s[3:5], s[5:7])])

# mpl.colors.ColorConverter.colors['r'] = hex2color('#e41a1c')
# mpl.colors.ColorConverter.colors['b'] = hex2color('#377eb8')
# mpl.colors.ColorConverter.colors['g'] = hex2color('#4daf4a')
# mpl.colors.ColorConverter.colors['p'] = hex2color('#984ea3')
# mpl.colors.ColorConverter.colors['y'] = hex2color('#ff7f00')


# Default color cycle for plot lines
# plt.rc('axes', color_cycle=['e41a1c', '377eb8', '4daf4a', '984ea3', 'ff7f00', 'ffff33']);

# Ben's color scheme taken from Tol's Palette 1:
#       tats.stackexchange.com/questions/118033/
#       best-series-of-colors-to-use-for-differentiating-series-in-publication-quality
# plt.rc('axes', color_cycle=['332288', '88ccee', '44aa99', '117733', '999933', 'ddcc77', 'cc6677', '882255', 'aa4499']);


### Lines
# plt.rc('lines', antialiased=True) # render lines in antialised (no jaggies)


### AXES
# plt.rc('axes', facecolor='eeeeee')     # axes background color
# plt.rc('axes', edgecolor='bcbcbc')     # axes edge color
# plt.rc('axes', linewidth=2)            # edge linewidth
# plt.rc('axes', grid=True)              # display grid or not
# plt.rc('axes', titlesize='medium')      # fontsize of the axes title [x-large]
# plt.rc('axes', labelsize='small')     # fontsize of the x any y labels [large]
# plt.rc('axes', labelcolor='111111')    # font color of labels
# plt.rc('axes', axisbelow=True)         # whether axis gridlines and ticks are below the axes elements (lines, text, etc)


### TICKS
# see http://matplotlib.sourceforge.net/api/axis_api.html#matplotlib.axis.Tick
# plt.rc('xtick.major', size=0)      # major tick size in points
# plt.rc('xtick.minor', size=0)      # minor tick size in points
# plt.rc('xtick.major', pad=6)       # distance to major tick label in points
# plt.rc('xtick.minor', pad=6)       # distance to the minor tick label in points
# plt.rc('xtick', color='111111')    # color of the tick labels
# plt.rc('xtick', direction='in')    # direction: in or out

# plt.rc('ytick.major', size=0)      # major tick size in points
# plt.rc('ytick.minor', size=0)      # minor tick size in points
# plt.rc('ytick.major', pad=6)       # distance to major tick label in points
# plt.rc('ytick.minor', pad=6)       # distance to the minor tick label in points
# plt.rc('ytick', color='111111')    # color of the tick labels
# plt.rc('ytick', direction='in')    # direction: in or out


### GRIDS
# plt.rc('grid', color='black')
# plt.rc('grid', linestyle=':')
# plt.rc('grid', linewidth=0.5)


### Legend
# plt.rc('legend', fancybox=True)  # if True, use a rounded box for the legend, else a rectangle
# legend.isaxes        : True
# legend.numpoints     : 2      # the number of points in the legend line
# legend.fontsize      : large  medium small
# legend.pad           : 0.0    # deprecated; the fractional whitespace inside the legend border
# legend.borderpad     : 0.5    # border whitspace in fontsize units
# legend.markerscale   : 1.0    # the relative size of legend markers vs. original
##the following dimensions are in axes coords
# legend.labelsep      : 0.010  # the vertical space between the legend entries
# legend.handlelen     : 0.05   # the length of the legend lines
# legend.handletextsep : 0.02   # the space between the legend line and legend text
# legend.axespad       : 0.02   # the border between the axes and legend edge
# legend.shadow        : False
