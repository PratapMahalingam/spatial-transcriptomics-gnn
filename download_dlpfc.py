"""Download the 12 LIBD human DLPFC Visium sections and their manual layer labels.

Sources (public, no login):
* count matrices: Lieber Institute AWS bucket listed in the HumanPilot README
* spot positions and barcode-level layer labels: github.com/LieberInstitute/HumanPilot
About 140 MB in total.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from stgnn.data import SAMPLE_DONOR

S3 = "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/{s}_filtered_feature_bc_matrix.h5"
GH = "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X/"


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  get  {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=Path("data"))
    args = ap.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)
    fetch(GH + "barcode_level_layer_map.tsv", args.dest / "barcode_level_layer_map.tsv")
    for s in SAMPLE_DONOR:
        fetch(S3.format(s=s), args.dest / f"{s}_filtered_feature_bc_matrix.h5")
        fetch(GH + f"{s}/tissue_positions_list.txt", args.dest / f"{s}_tissue_positions_list.txt")


if __name__ == "__main__":
    main()
