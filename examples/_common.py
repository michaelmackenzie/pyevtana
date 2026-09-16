"""Shared argument handling for the examples."""

import argparse

DEFAULT_FILE = "/exp/mu2e/app/users/mmackenz/main/nts.owner.description.version.sequencer.root"


def parse(description, **extra):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", nargs="*", default=[DEFAULT_FILE],
                        help="EventNtuple file(s), glob, or .txt filelist")
    for name, kwargs in extra.items():
        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
    args = parser.parse_args()
    if not args.files:
        args.files = [DEFAULT_FILE]
    return args
