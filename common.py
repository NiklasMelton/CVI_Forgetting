import os
import re
from glob import glob
from typing import Union, List


# helper to parse seed integer from filename
def _seed_num(fp):
    m = re.search(r"_seed_(\d+)", fp)
    return int(m.group(1)) if m else float("inf")

def find_lowest_seeded(files):
    # pick lowest-seed file for each
    return min(files, key=_seed_num)

def find_all_seeded(base_path: str) -> List[str]:
    base_root, ext = os.path.splitext(base_path)
    # if ext is empty, this still works as “…_seed_*”
    pattern = f"{base_root}_seed_*{ext}"
    fnames = sorted(glob(pattern))
    if not fnames:
        raise FileNotFoundError(f"No files matching {pattern}")
    return fnames

def find_seeded(filepath_or_list: Union[str, List[str], tuple]) -> Union[
    str, List[str]]:
    """
    Given either
      - a single unseeded filepath like ".../data.npz"
      - a list/tuple of such filepaths
    this returns the corresponding seeded variant(s), using the smallest integer
    seed that exists on disk (common to all, if you passed multiple).

    If you pass a string, you get back a string.
    If you pass a list/tuple, you get back a list of strings (same order).
    """

    def _get_seeds_for(fp):
        base_dir, filename = os.path.split(fp)
        name, ext = os.path.splitext(filename)
        pattern = os.path.join(base_dir, f"{name}_seed_*{ext}")
        regex = re.compile(rf"^{re.escape(name)}_seed_(\d+){re.escape(ext)}$")
        seed_map = {}
        for full in glob(pattern):
            m = regex.match(os.path.basename(full))
            if m:
                seed = int(m.group(1))
                seed_map[seed] = full
        if not seed_map:
            raise FileNotFoundError(
                f"No seeded file found matching '{filename}' in {base_dir!r}")
        return seed_map

    # single-file case
    if isinstance(filepath_or_list, str):
        seed_map = _get_seeds_for(filepath_or_list)
        smallest = min(seed_map)
        return seed_map[smallest]

    # multi-file case
    if isinstance(filepath_or_list, (list, tuple)):
        # build each file’s seed→path map and collect the seed sets
        maps = []
        seed_sets = []
        for fp in filepath_or_list:
            smap = _get_seeds_for(fp)
            maps.append(smap)
            seed_sets.append(set(smap))
        # find common seeds
        common = set.intersection(*seed_sets)
        if not common:
            raise FileNotFoundError(
                "No single seed has all files: " +
                ", ".join(os.path.basename(fp) for fp in filepath_or_list)
            )
        chosen = min(common)
        # reconstruct in same order
        return [m[chosen] for m in maps]

    # unsupported type
    raise TypeError("Expected a filepath string or a list/tuple of strings")


def make_dirs(path):
    dir_path = os.path.dirname(path)

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)