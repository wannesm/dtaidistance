#!/usr/bin/env python3
# encoding: utf-8
"""
check_installation.py

Created by Wannes Meert
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import sys

try:
    import dtaidistance
except ImportError as exc:
    print("Cannot import dtaidistance")
    sys.exit(1)

print('Location of dtaidistance:')
print(dtaidistance)

is_complete = dtaidistance.dtw.try_import_c(True)
if not is_complete:
    sys.exit(1)

sys.exit(0)

