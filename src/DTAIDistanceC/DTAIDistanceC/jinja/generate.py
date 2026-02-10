#!/usr/bin/env python3
# encoding: utf-8
"""
generate.py

Created by Wannes Meert on 19-08-2022.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
import jinja2


logger = logging.getLogger(__name__)
templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)


targets = {
    "dtw_distances_ptrs.c":
        ["dtw_distances.jinja.c",
            {'suffix': 'ptrs', 'nb_size': 'nb_ptrs'},
            []],
    "dtw_distances_matrix.c":
        ["dtw_distances.jinja.c",
            {'suffix': 'matrix', 'nb_size': 'nb_rows'},
            []],
    "dd_dtw.c":
        ["dd_dtw.jinja.c",
            {},
            ["dtw_distance.jinja.c", "dtw_distances.jinja.c",
             "dtw_warpingpaths.jinja.c", "dtw_dba.jinja.c",
             "dtw_expandwps.jinja.c", "dtw_bestpath.jinja.c",
             "lb_keogh.jinja.c", "dtw_dtwh.jinja.c"]],
    "dd_dtw_openmp.c":
        ["dd_dtw_openmp.jinja.c",
            {},
            ["dtw_distances_parallel.jinja.c"]],
    "dd_ed.c":
        ["dd_ed.jinja.c",
            {},
            ["ed_distance.jinja.c"]],
    "dd_loco.c":
        ["dd_loco.jinja.c",
            {},
            ["loco_warpingpaths.jinja.c", "loco_best_path.jinja.c"]],
}
essential_targets = ['dd_dtw.c', 'dd_dtw_openmp.c', 'dd_ed.c', 'dd_loco.c']


def generate(target):
    logger.info(f'Generating: {target}')
    fno = target
    fni, kwargs, _deps = targets[target]
    template = templateEnv.get_template(fni)
    outputText = template.render(**kwargs)
    with open(fno, 'w') as o:
        o.write(outputText)


def dependencies(target):
    logger.info(f'Dependencies for: {target}')
    fni, _kwargs, deps = targets[target]
    return [fni] + deps


def main(argv=None):
    parser = argparse.ArgumentParser(description='Perform some task')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('--deps', '-d', action='store_true', help='Print dependencies')
    parser.add_argument('--targets', '-t', action='store_true', help='Print targets')
    parser.add_argument('--all', '-a', action='store_true', help='Use all targets')
    # parser.add_argument('--output', '-o', required=True, help='Output file')
    # parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('input', nargs='*', help='List of target files to generate')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if args.input is None or len(args.input) == 0:
        inputs = essential_targets
    elif args.input[0] == "all":
        inputs = targets.keys()
    else:
        inputs = args.input

    if args.deps:
        deps = []
        for target in inputs:
            deps += dependencies(target)
        print(' '.join(deps))
        return 0

    if args.targets:
        if args.all:
            print(' '.join(targets.keys()))
        else:
            print(' '.join(inputs))
        return 0

    for target in inputs:
        generate(target)


if __name__ == "__main__":
    sys.exit(main())

