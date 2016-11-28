from . import dtw
try:
    from . import dtw_c
except ImportError:
    print("DTW C variant not available, run `make build` in the dtaidistance directory")
    dtw_c = None

__version__ = "0.1.0"
