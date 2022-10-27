import os
from argparse import Namespace, ArgumentParser
from typing import Tuple

from src.lib.io import readJSON
from src.lib.paths import getPath

EXPECTED_SIZE = 26


def main() -> None:
    args = parseArgs()

    path = getPath(os.path.join(args.directory, str(args.input_loadout)), "plastic-results")
    data = readJSON(path)

    for tag in "behavior_distribution", "goals":
        sizes, problematic_indices = getSizes(data, tag)
        print(f"{tag} has sizes {sizes}")
        assert sizes == {EXPECTED_SIZE}, f"Got: {sizes}; expected: {{{EXPECTED_SIZE}}}. "\
                                         f"\nProblematic indices are: {problematic_indices}"


def parseArgs() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-D", "--directory", type=str, required=True)
    parser.add_argument("-i", "--input-loadout", type=str, required=True)
    return parser.parse_args()


def getSizes(data: dict, tag: str, expected_size: int = EXPECTED_SIZE) -> Tuple[set, list]:
    sizes = set()
    problematic_indices = []
    for i, d in data.items():
        size = len(d[tag])
        if size != EXPECTED_SIZE:
            problematic_indices.append(i)
        sizes.add(len(d[tag]))
    return sizes, problematic_indices


if __name__ == "__main__":
    main()
