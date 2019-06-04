# -*- coding: UTF-8 -*-
"""
dtaidistance
~~~~~~~~~~~~

Time series distance methods.

:author: Wannes Meert
:copyright: Copyright 2017 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging


logger = logging.getLogger("be.kuleuven.dtai.distance")


from . import dtw
try:
    from . import dtw_c
except ImportError:
    # Try to compile automatically
    # try:
    #     import numpy as np
    #     import pyximport
    #     pyximport.install(setup_args={'include_dirs': np.get_include()})
    #     from . import dtw_c
    # except ImportError:
    # logger.warning("\nDTW C variant not available.\n\n" +
    #                "If you want to use the C libraries (not required, depends on cython), " +
    #                "then run `cd {};python3 setup.py build_ext --inplace`.".format(dtaidistance_dir))
    dtw_c = None

__version__ = "1.2.2"
__author__ = "Wannes Meert"
__copyright__ = "Copyright 2017-2019 KU Leuven, DTAI Research Group"
__license__ = "Apache License, Version 2.0"
