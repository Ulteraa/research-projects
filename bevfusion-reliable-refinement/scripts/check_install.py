"""Verify that the custom modules are importable and registered."""

from mmdet3d.registry import MODELS

import projects.BEVFusion.bevfusion  # noqa: F401


def main() -> None:
    expected = [
        'ReliabilityAwareFuser',
        'ObjectRefineTransFusionHead',
    ]
    missing = [name for name in expected if MODELS.get(name) is None]
    if missing:
        raise RuntimeError(f'Missing registry entries: {missing}')
    print('Custom BEVFusion modules are registered:', ', '.join(expected))


if __name__ == '__main__':
    main()
