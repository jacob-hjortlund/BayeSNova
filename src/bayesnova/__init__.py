import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

import bayesnova.analysis as analysis
import bayesnova.calibration as calibration
import bayesnova.cosmo as cosmo
import bayesnova.progenitors as progenitors
import bayesnova.mixture as mixture
import bayesnova.base as base
import bayesnova.plot as plot
import bayesnova.ladder as ladder

import autoconf.conf as conf

conf.instance.register(__file__)
