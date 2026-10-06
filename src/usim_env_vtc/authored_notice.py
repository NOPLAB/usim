"""Carry geographic provenance separately from original artwork/software licensing."""

import argparse
from pathlib import Path
from typing import Final


NOTICE: Final = """Original-art VTC geographic scene

Original procedural artwork and tooling: Copyright (c) 2026 nop.
Recipe software and independently authored recipe artwork: MIT; see LICENSE-MIT.
The assembled map is NOT blanket MIT or established commercially unrestricted.

Retained VTC map metadata/BSP:
Copyright 2017-2020 Tomoaki Yoshida <yoshida@furo.org>.
https://github.com/furo-org/VTC
Revision c503a8827cb1134a4931fd93cb1af093df5ba35f.
Otherwise-unspecified VTC-owned contributions are Apache-2.0; see LICENSE-APACHE2.
Imported original City Hall/furniture dimensions retain their source provenance;
their model originals are declared CC BY-NC-SA 4.0. Metadata is not relicensed here.

Retained landscape heights/masks:
GSI basic map/5 m DEM sheets 544000/544010, approval R 2JHs 231,
and VTC/fuRo 2020 measurement contribution.
https://www.gsi.go.jp/LAW/2930-qa.html
Unresolved downstream landscape reuse questions remain.

Inferred tree and column placement:
fuRo, official Tsukuba Challenge 2019 PCD, CC BY-NC-SA 4.0.
https://github.com/tsukubachallenge/tc-datasets
https://creativecommons.org/licenses/by-nc-sa/4.0/
Requested citation: Yoshitaka Hara and Masahiro Tomono, Moving Object Removal
and Surface Mesh Mapping for Path Planning on 3D Terrain,
Advanced Robotics 34(6), 375-387, 2020.
https://doi.org/10.1080/01691864.2020.1717375

Changes: original model/texture artwork replaced by independent parametric
recipes fitted to retained dimensional envelopes; source actor placement
retained; road-relief correction retained; original surface colors generated
using geographic land-cover masks; raw survey display/helper art omitted.
All original source files remain unmodified. No survey accuracy is asserted.
No Epic/KiteDemo/Paragon/StarterContent/Megascans model or texture bytes are
included or referenced. No artwork-pack entitlement is conferred by this notice.
See docs/vtc/authored-models.md for distinctions and source audit links.
"""


def write_notice(out: Path, software_license: Path, vtc_license: Path) -> None:
    """Include notices and the applicable license copies without changing their scope."""
    (out / 'NOTICE').write_text(NOTICE, encoding='utf-8')
    (out / 'LICENSE-MIT').write_bytes(software_license.read_bytes())
    (out / 'LICENSE-APACHE2').write_bytes(vtc_license.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--software-license', type=Path, default=Path('LICENSE'))
    parser.add_argument(
        '--vtc-license', type=Path, default=Path('assets/vtc-full/upstream/VTC/LICENSE')
    )
    args = parser.parse_args()
    write_notice(args.out, args.software_license, args.vtc_license)


if __name__ == '__main__':
    main()
