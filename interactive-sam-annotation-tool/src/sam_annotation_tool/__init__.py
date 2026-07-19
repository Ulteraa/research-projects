"""Interactive SAM annotation tool."""

from .coco import CocoDataset, bbox_from_polygons, polygon_area, validate_document

__all__ = ["CocoDataset", "bbox_from_polygons", "polygon_area", "validate_document"]
__version__ = "0.1.0"
