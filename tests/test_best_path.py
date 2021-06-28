from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import matplotlib.pyplot as plt

import random
x = np.arange(0, 20, .5)
s1 = np.sin(x)
s2 = np.sin(x - 1)
random.seed(1)
for idx in range(len(s2)):
    if random.random() < 0.05:
        s2[idx] += (random.random() - 0.5) / 2
d, paths = dtw.warping_paths(s1, s2, window=25, psi=30) # adjust psi here for relaxation
best_path = dtw.best_path(paths)

best_path2 = dtw.best_path2(paths) # infinite loop in the old version because the path gets stuck at the left side !

dtwvis.plot_warpingpaths(s1, s2, paths, best_path)
plt.show()

dtwvis.plot_warpingpaths(s1, s2, paths, best_path2)
plt.show()