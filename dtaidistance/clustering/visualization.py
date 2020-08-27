

from ..exceptions import MatplotlibException


def prepare_plot_options(show_ts_label, show_tr_label):
    if show_ts_label is True:
        show_ts_label = lambda idx: str(int(idx))
    elif show_ts_label is False or show_ts_label is None:
        show_ts_label = lambda idx: ""
    elif callable(show_ts_label):
        pass
    elif hasattr(show_ts_label, "__getitem__"):
        show_ts_label_prev = show_ts_label
        show_ts_label = lambda idx: show_ts_label_prev[idx]
    else:
        raise AttributeError("Unknown type for show_ts_label, expecting boolean, subscriptable or callable, "
                             "got {}".format(type(show_ts_label)))

    if show_tr_label is True:
        show_tr_label = lambda dist: "{:.2f}".format(dist)
    elif show_tr_label is False or show_tr_label is None:
        show_tr_label = lambda dist: ""
    elif callable(show_tr_label):
        pass
    elif hasattr(show_tr_label, "__getitem__"):
        show_tr_label_prev = show_tr_label
        show_tr_label = lambda idx: show_tr_label_prev[idx]
    else:
        raise AttributeError("Unknown type for show_ts_label, expecting boolean, subscriptable or callable, "
                             "got {}".format(type(show_ts_label)))

    return show_ts_label, show_tr_label
