import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import src.bayesnova.analysis as analysis
import src.bayesnova.calibration as calibration
import src.bayesnova.cosmo as cosmo
import src.bayesnova.volumetric_rate as volumetric_rate
import src.bayesnova.mixture as mixture
import src.bayesnova.base as base

import autoconf.conf as conf
conf.instance.register(__file__)