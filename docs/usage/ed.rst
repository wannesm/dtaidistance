Euclidean Distance (ED)
~~~~~~~~~~~~~~~~~~~~~~~

::

    from dtaidistance import ed
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
    s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
    distance = ed.distance(s1, s2)
    print(distance)


Different lenghts
"""""""""""""""""

The Euclidean distance also handles sequences of different lengths by
comparing the last element of the shortest series to the remaining
elements in the longer series. This is compatible with Euclidean
distance being used as an upper bound for DTW.
