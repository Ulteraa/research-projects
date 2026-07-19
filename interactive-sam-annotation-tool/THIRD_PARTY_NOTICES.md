# Third-party notices and provenance

This document records material needed to run the curated project and provenance
observed in the uploaded research archive. It is not legal advice.

## Segment Anything

The archive contained a complete copy of Meta's Segment Anything repository.
The copied files match upstream commit
`7fa17d78c45f4f642faa89f4c0e590c400f74225` (2023-04-10). This curated project
does not redistribute that source; `requirements.txt` pins the same commit.

- Project: <https://github.com/facebookresearch/segment-anything>
- License: Apache License 2.0
- Copyright: Meta Platforms, Inc. and affiliates

SAM checkpoints are not included. Obtain weights from the official project and
review the terms distributed with them.

## Runtime libraries

The project uses PyTorch, NumPy, Pillow, Matplotlib, and OpenCV. Those packages
retain their own licenses and notices. Installing this project does not change
their license terms.

## Uploaded archive license ambiguity

The archive's root `LICENSE` is an MIT license naming Hu Ye and appears to have
been inherited from the bundled Swin Transformer material. The archive did not
include a separate, clearly attributable license grant for its custom
`inetractive_openCV_Edit.py` annotation script. The same supplied license text
is preserved as `LICENSE-ARCHIVE` for provenance, but it is not a project-wide
license and should not be treated as definitive proof that all original custom
code was released under those terms.

Confirm the custom code's license with the project author before redistribution
or commercial use. Any reuse must also retain the applicable SAM and dependency
notices.
