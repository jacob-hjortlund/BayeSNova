import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import bayesnova.analysis as analysis
import bayesnova.calibration as calibration
import bayesnova.cosmo as cosmo
import bayesnova.volumetric_rate as volumetric_rate
import bayesnova.mixture as mixture
import bayesnova.base as base

import autoconf.conf as conf
conf.instance.register(__file__)