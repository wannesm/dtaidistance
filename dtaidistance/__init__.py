import logging
logger = logging.getLogger("be.kuleuven.dtai.distance")

from . import dtw
try:
    from . import dtw_c
except ImportError:
    import os
    # Try to compile automatically
    # try:
    #     import numpy as np
    #     import pyximport
    #     pyximport.install(setup_args={'include_dirs': np.get_include()})
    #     from . import dtw_c
    # except ImportError:
    dtaidistance_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)
    logger.warning("\nDTW C variant not available.\n\n" +
                   "If you want to use the C libraries (not required, depends on cython), " +
                   "then run `cd {};python3 setup.py build_ext --inplace`.".format(dtaidistance_dir))
    dtw_c = None

__version__ = "0.1.1"
