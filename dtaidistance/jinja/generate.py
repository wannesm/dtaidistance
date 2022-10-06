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



seq_t = "float"
seq_tpy = "float"
# Also change the type in lib/DTAIDistanceC/DTAIDistanceC/dd_globals.h


targets = {
    "dtw_cc.pyx":
        ["dtw_cc.jinja.pyx",
            {"seq_tpy": seq_tpy, "seq_t": seq_t},
            ["dtw_cc_warpingpaths.jinja.pyx",
             "dtw_cc_distancematrix.jinja.pyx",
             "dtw_cc_warpingpath.jinja.pyx",
             "dtw_cc_dba.jinja.pyx"]],
    "dtw_cc_omp.pyx":
        ["dtw_cc_omp.jinja.pyx",
            {"seq_tpy": seq_tpy, "seq_t": seq_t},
            []],
    "dtw_cc.pxd":
        ["dtw_cc.jinja.pxd",
            {"seq_tpy": seq_tpy, "seq_t": seq_t},
            []],
    "dtaidistancec_globals.pxd":
        ["dtaidistancec_globals.jinja.pxd",
            {"seq_tpy": seq_tpy, "seq_t": seq_t},
            []],
    "ed_cc.pyx":
        ["ed_cc.jinja.pyx",
            {"seq_tpy": seq_tpy, "seq_t": seq_t},
            []],
}
essential_targets = ['dtw_cc.pyx', 'dtw_cc.pxd', 'dtaidistancec_globals.pxd']


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
    parser = argparse.ArgumentParser(description='Generate source code files from templates')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('--deps', '-d', action='store_true', help='Print dependencies')
    parser.add_argument('--targets', '-t', action='store_true', help='Print available targets')
    # parser.add_argument('--output', '-o', required=True, help='Output file')
    # parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('input', nargs='*', help='List of target files to generate')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if args.targets:
        print('Targets:')
        for k in targets.keys():
            e = ' (default)' if k in essential_targets else ''
            print(f'- {k}{e}')
        return 0

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

    for target in inputs:
        generate(target)


if __name__ == "__main__":
    sys.exit(main())


