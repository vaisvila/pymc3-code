%matplotlib inline
import logging

from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pymc3 as pm
from pymc3.distributions.dist_math import binomln, bound, factln
import scipy as sp
import seaborn as sns
import theano
from theano import tensor as tt

sns.set()

PCT_FORMATTER = StrMethodFormatter('{x:.1%}')
SEED = 518302 # from random.org, for reproducibility

np.random.seed(SEED)
# keep theano from complaining about compile locks
(logging.getLogger('theano.gof.compilelock')
        .setLevel(logging.CRITICAL))

# keep theano from warning about default rounding mode changes
theano.config.warn.round = False