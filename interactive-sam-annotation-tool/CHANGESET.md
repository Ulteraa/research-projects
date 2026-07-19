# Curation changes

This folder reconstructs the research contribution from the uploaded archive as
a focused project. It is not a byte-for-byte repackaging of the original tree.

## Preserved behavior

- SAM mask generation from box prompts.
- SAM mask generation from foreground point prompts.
- Manual polygon placement.
- Saved-annotation clearing/undo concepts and COCO polygon export.
- Visual review of the exported annotation result.

## Reliability and usability changes

- Added explicit command-line paths for images, output, categories, model type,
  checkpoint, and device; removed personal absolute paths.
- Load SAM once per session instead of once per image.
- Keep the original image unchanged when setting each SAM predictor image.
- Added negative point prompts and multi-category configuration.
- Write one resumable dataset-level COCO JSON file with atomic replacement.
- Corrected the archive's reversed image width/height fields and string IDs.
- Compute annotation area from the mask/polygon instead of bounding-box area.
- Make undo image-scoped and make erase update the saved COCO geometry.
- Added validation, standalone Pillow visualization, tests, packaging metadata,
  requirements, documentation, and an explicit limitations record.

## Deliberately excluded archive content

- The copied full Segment Anything repository, its notebooks, demo assets, and
  cached files. The exact upstream commit is installed as a dependency instead.
- Duplicate SAM/Swin model implementations at multiple archive paths.
- Detectron2 training and configuration material not required by the annotation
  application.
- `__pycache__`, `.pyc`, downloaded ZIP files, transfer logs, generated images,
  and local machine paths.
- A Detectron2-only visualization path; the curated renderer uses Pillow and
  consumes ordinary COCO polygons directly.

## Known semantic difference

The original screenshot-oriented script mixed display edits and annotation
state. In the curated version, `clear` only discards the unsaved candidate,
`undo` removes the last saved object for the current image, and `erase` modifies
saved polygon geometry. This makes the on-screen state and the persisted COCO
file agree.
