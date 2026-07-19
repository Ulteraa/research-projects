# Results and validation status

## Evidence supplied with the project

The seven-page report, *Interactive Annotation and Segmentation Tool Using
Segment Anything Model*, demonstrates the intended workflow with screenshots:

- a rectangular SAM prompt producing a human mask;
- one or more positive point prompts producing a human mask;
- manual polygon placement;
- clearing, undoing, brushing/erasing, and saving annotations;
- COCO polygon JSON output; and
- visual inspection of saved annotations.

These examples establish a qualitative proof of concept. They do **not** provide
timings, annotation throughput, mask IoU, inter-annotator agreement, a user
study, or a controlled comparison with another annotation tool. Consequently,
neither the report nor this curated implementation supports a quantitative
speedup or accuracy claim.

## Validation performed for this curated version

The local dependency-light checks cover:

- polygon area and XYWH box computation;
- integer COCO image and annotation IDs;
- correct width/height ordering;
- resumable IDs and category-consistency checks;
- image-scoped undo behavior;
- detection of broken annotation references;
- nested-path visualization; and
- Python bytecode compilation and command-line parser startup.

Run the same checks with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
PYTHONPATH=src python -m sam_annotation_tool.app --help
```

At curation time, all seven unit tests passed. SAM inference was not executed in
the curation environment because PyTorch, OpenCV, the SAM package, a model
checkpoint, and a graphical display were not present. A machine with those
runtime dependencies should therefore complete a brief end-to-end acceptance
test before production annotation begins.

## Suggested acceptance test

1. Annotate one image once with a box, mixed positive/negative points, and a
   manual polygon.
2. Save each candidate, erase part of one saved region, undo another, and close
   the window.
3. Restart the command against the same output file and confirm the saved
   objects reappear with stable IDs.
4. Run `sam-validate-coco` with `--image-root`.
5. Render the file with `sam-visualize-coco` and visually inspect polygons,
   category labels, object boundaries, and small/disconnected regions.

For a publishable comparison, record per-image annotation time, correction
counts, final mask IoU against an independently reviewed reference set, and
results from multiple annotators. Fix the image set, hardware, checkpoint,
category configuration, and tool versions before comparing methods.
