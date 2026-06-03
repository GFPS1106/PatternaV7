# Patterna Parameter Guide

This guide explains the main user-adjustable parameters in `src/PatternaV7.py`.

Most settings are located near the top of the script. Users should adjust these parameters before running Patterna on a new dataset.

---

## 1. Input and output paths

```python
INPUT_DIR = r"path/to/input/images"
OUT_DIR = r"path/to/output/folder"
```

### `INPUT_DIR`

Folder containing the TIFF or OME-TIFF images to analyze.

Patterna will search this folder for microscopy image files.

### `OUT_DIR`

Folder where all Patterna outputs will be saved.

If the folder does not exist, Patterna will create it.

---

## 2. Z-stack handling

```python
Z_MODE = "max"
```

Controls how Z-stacks are converted into 2D images.

Options:

```text
"max"   = maximum-intensity projection
"mean"  = mean-intensity projection
"mid"   = middle Z-slice
integer = specific Z-slice index, using Python 0-based indexing
```

Recommended default:

```python
Z_MODE = "max"
```

Use `"max"` for most immunofluorescence images unless there is a reason to preserve only one optical section.

---

## 3. Channel names

```python
CHANNEL_NAMES = {
    0: "DAPI",
    1: "TBXT",
    2: "SOX2",
    3: "GATA6",
}
```

Defines readable names for each cleaned channel.

Current default mapping:

```text
clean ch0 = DAPI
clean ch1 = TBXT
clean ch2 = SOX2
clean ch3 = GATA6
```

If the marker cocktail changes, update this dictionary.

Example:

```python
CHANNEL_NAMES = {
    0: "DAPI",
    1: "TFAP2C",
    2: "OCT4",
    3: "GATA3",
}
```

Important: the channel numbers must match the cleaned image stack used by Patterna.

---

## 4. Brightfield channel removal

Patterna currently includes logic for 5-channel images:

```python
if img.shape[0] == 5:
    img = img[[0, 2, 3, 4], :, :]
```

This removes the brightfield channel and keeps the fluorescence channels.

Current assumption:

```text
original ch0 = DAPI
original ch1 = brightfield
original ch2 = marker 1
original ch3 = marker 2
original ch4 = marker 3
```

After cleaning:

```text
clean ch0 = DAPI
clean ch1 = marker 1
clean ch2 = marker 2
clean ch3 = marker 3
```

Always check the channel QC images in `debug/` to confirm that the channel mapping is correct.

---

## 5. Colony mask settings

```python
MASK_MODE = "dapi"
DAPI_CHANNEL = 0
THRESH_METHOD = "percentile"
THRESH_PCT = 70
```

These settings control how the whole-colony mask is generated.

### `MASK_MODE`

Options:

```text
"dapi"         = use the DAPI channel to build the colony mask
"composite"    = use a composite of all channels
"dapi_sauvola" = use a more sensitive DAPI-based local thresholding approach
```

Recommended default:

```python
MASK_MODE = "dapi"
```

Use `"composite"` if DAPI is weak but marker signal clearly outlines the colony.

Use `"dapi_sauvola"` if the DAPI signal is uneven and the normal mask fails.

### `DAPI_CHANNEL`

Usually:

```python
DAPI_CHANNEL = 0
```

Change this only if DAPI is not in channel 0.

### `THRESH_METHOD`

Options:

```text
"percentile" = threshold based on intensity percentile
"otsu"       = automatic Otsu threshold
```

Recommended default:

```python
THRESH_METHOD = "percentile"
```

### `THRESH_PCT`

Used only when `THRESH_METHOD = "percentile"`.

```python
THRESH_PCT = 70
```

How to tune:

```text
Higher value = stricter mask, less area included
Lower value  = more permissive mask, more area included
```

If the mask is too small, decrease `THRESH_PCT`.

If the mask includes too much background, increase `THRESH_PCT`.

---

## 6. Spiderweb sampling settings

```python
N_ANGLES = 240
N_R = 200
SIGMA_R = 2.5
```

These parameters control the spatial sampling grid.

### `N_ANGLES`

Number of angular directions sampled from the colony centroid to the boundary.

```python
N_ANGLES = 240
```

Higher values give smoother angular resolution but slightly larger files and longer runtime.

Recommended range:

```text
120–240 for most analyses
240+ for high-resolution spatial maps
```

### `N_R`

Number of radial points sampled along each angular direction.

```python
N_R = 200
```

Higher values give smoother radial profiles.

Recommended range:

```text
100–200 for most analyses
```

### `SIGMA_R`

Amount of smoothing applied along the radial direction.

```python
SIGMA_R = 2.5
```

Higher values produce smoother profiles but can blur sharp spatial transitions.

Recommended range:

```text
1.0–3.0
```

---

## 7. Output toggles

```python
SAVE_HEATMAPS = True
SAVE_DEBUG = True
SAVE_GRIDS = True
SAVE_MASKS = True
```

### `SAVE_HEATMAPS`

Saves rectangular angle-radius heatmaps.

Recommended:

```python
SAVE_HEATMAPS = True
```

### `SAVE_DEBUG`

Saves debug overlays, including spiderweb sampling overlays.

Recommended:

```python
SAVE_DEBUG = True
```

Keep this on unless the pipeline is already fully validated.

### `SAVE_GRIDS`

Saves the core spiderweb sampled arrays as `.npy` files.

Recommended:

```python
SAVE_GRIDS = True
```

These are needed for grouped heatmaps and some downstream analyses.

### `SAVE_MASKS`

Saves binary colony masks.

Recommended:

```python
SAVE_MASKS = True
```

These are important for QC and grouped orthogonal analysis.

---

## 8. Grouped output settings

```python
EXPORT_CHANNEL_PROFILE_WORKBOOK = True
EXPORT_GROUPED_HEATMAPS = True
EXPORT_GROUPED_ORTHOGONAL = True
EXPORT_INTEGRATED_SUMMARY_PANELS = True
```

### `EXPORT_CHANNEL_PROFILE_WORKBOOK`

Exports Excel workbooks with radial profiles organized by channel.

Recommended:

```python
EXPORT_CHANNEL_PROFILE_WORKBOOK = True
```

### `EXPORT_GROUPED_HEATMAPS`

Generates averaged rectangular and polar heatmaps across multiple micropatterns.

Recommended:

```python
EXPORT_GROUPED_HEATMAPS = True
```

Requires saved grids in `grids/`.

### `EXPORT_GROUPED_ORTHOGONAL`

Generates grouped major/minor axis profile plots.

Recommended:

```python
EXPORT_GROUPED_ORTHOGONAL = True
```

Requires saved masks and grids.

### `EXPORT_INTEGRATED_SUMMARY_PANELS`

Generates one integrated summary panel per image.

Recommended:

```python
EXPORT_INTEGRATED_SUMMARY_PANELS = True
```

These panels are useful for quick visual inspection and presentations, but they may take extra time to generate.

---

## 9. Profile grouping settings

```python
GROUP_PROFILE_REPLICATES = True
PROFILE_ERROR_MODE = "sd"
```

### `GROUP_PROFILE_REPLICATES`

Groups radial profiles from multiple micropatterns into channel-level summaries.

Recommended:

```python
GROUP_PROFILE_REPLICATES = True
```

### `PROFILE_ERROR_MODE`

Controls the variability shown in grouped profile plots.

Options:

```text
"sd"  = standard deviation
"sem" = standard error of the mean
```

Recommended default:

```python
PROFILE_ERROR_MODE = "sd"
```

Use `"sd"` to show variability between micropatterns.

Use `"sem"` to show uncertainty around the mean.

---

## 10. Single-cell analysis settings

```python
DO_CELLS = True
NUC_DAPI_CHANNEL = 0
N_SECTORS = 12
RING_EDGES = (0.33, 0.67)
```

### `DO_CELLS`

Turns nuclei segmentation and single-cell measurements on or off.

```python
DO_CELLS = True
```

Set to `False` if you only want colony-level spatial profiles and do not need cell-level outputs.

### `NUC_DAPI_CHANNEL`

Channel used for nuclei segmentation.

Usually:

```python
NUC_DAPI_CHANNEL = 0
```

### `N_SECTORS`

Number of angular sectors used for assigning cells to angular bins.

```python
N_SECTORS = 12
```

This is mainly useful for angular/spatial cell-position analysis.

### `RING_EDGES`

Defines the radial zones for cell classification.

```python
RING_EDGES = (0.33, 0.67)
```

This creates three radial zones:

```text
inner  = r_frac < 0.33
middle = 0.33 ≤ r_frac < 0.67
outer  = r_frac ≥ 0.67
```

---

## 11. Nuclei segmentation mode

```python
NUC_SEGMENTATION_MODE = "stardist"
```

Options:

```text
"stardist" = use StarDist nuclei segmentation
"hybrid"   = try StarDist first, then fall back to dense watershed if count is too low
```

Recommended default:

```python
NUC_SEGMENTATION_MODE = "stardist"
```

Use `"hybrid"` for very dense colonies where StarDist under-segments the interior.

---

## 12. StarDist settings

```python
STARDIST_MODEL_NAME = "2D_versatile_fluo"
STARDIST_PROB_THRESH = 0.30
STARDIST_NMS_THRESH = 0.55
STARDIST_MIN_SIZE = None
STARDIST_MAX_SIZE = None
STARDIST_BBOX_PAD = 30
```

### `STARDIST_MODEL_NAME`

Pretrained StarDist model used for nuclei segmentation.

```python
STARDIST_MODEL_NAME = "2D_versatile_fluo"
```

### `STARDIST_PROB_THRESH`

Probability threshold for StarDist object detection.

```python
STARDIST_PROB_THRESH = 0.30
```

How to tune:

```text
Lower value  = more nuclei detected, more permissive
Higher value = fewer nuclei detected, stricter
```

If StarDist is under-segmenting, decrease this value.

If StarDist is detecting too many false positives, increase this value.

Suggested range:

```text
0.10–0.50
```

### `STARDIST_NMS_THRESH`

Non-maximum suppression threshold.

```python
STARDIST_NMS_THRESH = 0.55
```

How to tune:

```text
Lower value  = stricter separation, may reduce overlapping detections
Higher value = allows more nearby/overlapping objects
```

If nuclei are being merged, try adjusting this together with `STARDIST_PROB_THRESH`.

Suggested range:

```text
0.30–0.70
```

### `STARDIST_MIN_SIZE` and `STARDIST_MAX_SIZE`

Area filtering after StarDist segmentation.

```python
STARDIST_MIN_SIZE = None
STARDIST_MAX_SIZE = None
```

Set to `None` to turn off area filtering.

This is useful when area filtering accidentally removes valid nuclei.

Example:

```python
STARDIST_MIN_SIZE = 25
STARDIST_MAX_SIZE = 300
```

Use area filtering only after checking the segmentation QC overlays.

### `STARDIST_BBOX_PAD`

Padding around the colony bounding box before running StarDist.

```python
STARDIST_BBOX_PAD = 30
```

Higher values include more surrounding area.

Lower values make segmentation faster but may crop edge nuclei if too small.

Recommended range:

```text
20–50
```

---

## 13. Hybrid segmentation safeguard

```python
HYBRID_MIN_ACCEPTABLE_NUCLEI = 800
```

Used only when:

```python
NUC_SEGMENTATION_MODE = "hybrid"
```

Patterna first runs StarDist. If the number of detected nuclei is below this threshold, Patterna switches to dense watershed fallback.

How to tune:

```text
Higher value = more likely to trigger watershed fallback
Lower value  = more likely to keep StarDist result
```

For dense 800 µm micropatterns, very low counts are biologically suspicious, so a high guardrail can be useful.

---

## 14. Dense watershed fallback settings

```python
DENSE_WS_MIN_AREA = 10
DENSE_WS_MAX_AREA = 500
DENSE_WS_GAUSS_SIGMA = 0.50
DENSE_WS_BG_SIGMA = 30
DENSE_WS_MIN_DISTANCE = 3
DENSE_WS_BLOCK_SIZE = 31
DENSE_WS_C = 0
```

Used only in hybrid mode when StarDist under-segments.

### `DENSE_WS_MIN_AREA` and `DENSE_WS_MAX_AREA`

Minimum and maximum object area retained after watershed.

If small real nuclei are being removed, decrease `DENSE_WS_MIN_AREA`.

If debris or tiny specks are counted, increase `DENSE_WS_MIN_AREA`.

If clumps are being included as cells, decrease `DENSE_WS_MAX_AREA`.

### `DENSE_WS_GAUSS_SIGMA`

Smoothing before thresholding.

Higher values smooth noise but can merge close nuclei.

### `DENSE_WS_BG_SIGMA`

Background correction scale.

Higher values correct slower background gradients.

### `DENSE_WS_MIN_DISTANCE`

Minimum distance between watershed seeds.

```text
Lower value  = more seeds, more splitting
Higher value = fewer seeds, less splitting
```

If nuclei are merged, decrease this value.

If nuclei are over-split, increase this value.

### `DENSE_WS_BLOCK_SIZE`

Adaptive thresholding window size.

Must be an odd number.

Larger values respond to broader intensity variation.

### `DENSE_WS_C`

Adaptive threshold offset.

Changing this affects how permissive the foreground detection is.

---

## 15. Cell-intensity map settings

```python
EXPORT_STANDALONE_CELL_MAPS = True
EXPORT_GROUPED_DISK_MARKER_MAPS = True

CELL_INTENSITY_MAP_STAT = "mean"
CELL_INTENSITY_MAP_BG = "white"
CELL_INTENSITY_NORM_MODE = "global_percentile"
CELL_INTENSITY_NORM_PCTS = (0.5, 99.5)
```

### `CELL_INTENSITY_MAP_STAT`

Cell intensity statistic used for display.

Options:

```text
"mean" = average marker intensity per cell
"p90"  = 90th percentile marker intensity per cell
```

Recommended:

```python
CELL_INTENSITY_MAP_STAT = "mean"
```

Use `"p90"` if the marker is spatially concentrated inside cells and the mean underestimates signal.

### `CELL_INTENSITY_MAP_BG`

Background color for cell maps.

Options:

```text
"white"
"black"
```

Recommended:

```python
CELL_INTENSITY_MAP_BG = "white"
```

### `CELL_INTENSITY_NORM_MODE`

Controls how cell-intensity maps are scaled for visualization.

Options:

```text
"per_image_percentile" = each image scaled separately
"per_image_minmax"     = each image scaled to its own min/max
"global_percentile"    = one scale per marker across all images
"manual"               = use manually defined limits
```

Recommended:

```python
CELL_INTENSITY_NORM_MODE = "global_percentile"
```

Use `"global_percentile"` for comparing multiple micropatterns.

Use `"per_image_percentile"` for exploratory visualization only.

### `CELL_INTENSITY_NORM_PCTS`

Percentiles used for global or per-image display scaling.

```python
CELL_INTENSITY_NORM_PCTS = (0.5, 99.5)
```

This helps avoid extreme outliers dominating the color scale.

---

## 16. Phenotype classification settings

```python
EXPORT_CELL_PHENOTYPES = True
PHENOTYPE_MARKER_CHANNELS = [1, 2, 3]
PHENOTYPE_THRESHOLD_MODE = "percentile_global"
PHENOTYPE_POSITIVE_PERCENTILE = 75
```

### `EXPORT_CELL_PHENOTYPES`

Turns cell phenotype classification outputs on or off.

```python
EXPORT_CELL_PHENOTYPES = True
```

### `PHENOTYPE_MARKER_CHANNELS`

Marker channels used for phenotype classification.

Usually DAPI is excluded.

```python
PHENOTYPE_MARKER_CHANNELS = [1, 2, 3]
```

### `PHENOTYPE_THRESHOLD_MODE`

Controls how marker-positive cells are defined.

Options:

```text
"manual"               = use explicit thresholds
"control"              = calculate thresholds from selected control images
"percentile_global"    = threshold from all cells pooled across images
"percentile_per_image" = threshold separately for each image
```

Recommended for exploration:

```python
PHENOTYPE_THRESHOLD_MODE = "percentile_global"
```

Recommended for final biological interpretation:

```python
PHENOTYPE_THRESHOLD_MODE = "control"
```

or

```python
PHENOTYPE_THRESHOLD_MODE = "manual"
```

with validated thresholds.

### `PHENOTYPE_POSITIVE_PERCENTILE`

Shared exploratory percentile cutoff.

```python
PHENOTYPE_POSITIVE_PERCENTILE = 75
```

Higher values make classification stricter.

Lower values make classification more permissive.

---

## 17. Marker-specific phenotype percentiles

```python
PHENOTYPE_POSITIVE_PERCENTILE_BY_CHANNEL = {
    1: 75,
    2: 60,
    3: 90,
}
```

Allows each marker to use a different positivity percentile.

This is useful when one marker is much brighter, dimmer, noisier, or more spatially widespread than others.

Example interpretation:

```text
TBXT ch1 = 75th percentile
SOX2 ch2 = 60th percentile
GATA6 ch3 = 90th percentile
```

Use higher percentiles for markers that are too permissive.

Use lower percentiles for markers that are too strict.

---

## 18. Manual phenotype thresholds

```python
PHENOTYPE_MANUAL_THRESHOLDS = {
    1: None,
    2: None,
    3: None,
}
```

Used only when:

```python
PHENOTYPE_THRESHOLD_MODE = "manual"
```

Example:

```python
PHENOTYPE_MANUAL_THRESHOLDS = {
    1: 75,
    2: 90,
    3: 60,
}
```

Manual thresholds must be in the same units as the selected cell intensity statistic.

---

## 19. Control-derived phenotype thresholds

```python
CONTROL_IMAGES_FOR_THRESHOLDS = ["CTRL", "control"]

CONTROL_IMAGES_FOR_THRESHOLDS_BY_CHANNEL = {
    1: [],
    2: [],
    3: [],
}

CONTROL_THRESHOLD_METHOD = "mean_plus_2sd"
```

Used only when:

```python
PHENOTYPE_THRESHOLD_MODE = "control"
```

### `CONTROL_IMAGES_FOR_THRESHOLDS`

Patterna searches image names for these substrings and uses the matching cells as control cells.

Example:

```python
CONTROL_IMAGES_FOR_THRESHOLDS = ["CTRL"]
```

### `CONTROL_IMAGES_FOR_THRESHOLDS_BY_CHANNEL`

Optional marker-specific control image names.

Example:

```python
CONTROL_IMAGES_FOR_THRESHOLDS_BY_CHANNEL = {
    1: ["TBXT_negative_control"],
    2: ["SOX2_negative_control"],
    3: ["GATA6_negative_control"],
}
```

### `CONTROL_THRESHOLD_METHOD`

Options:

```text
"mean_plus_2sd"
"mean_plus_3sd"
"percentile_95"
"percentile_99"
```

Recommended starting point:

```python
CONTROL_THRESHOLD_METHOD = "mean_plus_2sd"
```

Use stricter methods if control signal has high background.

---

## 20. Phenotype intensity statistic

```python
PHENOTYPE_INTENSITY_STAT = "mean"
```

Controls which per-cell intensity measurement is used for phenotype classification.

Options:

```text
"mean" = average intensity per cell
"p90"  = 90th percentile intensity per cell
```

Recommended:

```python
PHENOTYPE_INTENSITY_STAT = "mean"
```

Use `"p90"` if marker signal is localized or punctate.

---

## 21. Common troubleshooting

### The colony mask is too small

Try:

```python
THRESH_PCT = 60
```

or switch:

```python
MASK_MODE = "composite"
```

### The colony mask includes too much background

Try:

```python
THRESH_PCT = 80
```

### StarDist under-segments nuclei

Try:

```python
STARDIST_PROB_THRESH = 0.18
STARDIST_NMS_THRESH = 0.65
```

or use:

```python
NUC_SEGMENTATION_MODE = "hybrid"
```

### StarDist detects too many false positives

Try:

```python
STARDIST_PROB_THRESH = 0.40
```

or add area filtering:

```python
STARDIST_MIN_SIZE = 25
STARDIST_MAX_SIZE = 300
```

### Area filtering removes valid nuclei

Turn it off:

```python
STARDIST_MIN_SIZE = None
STARDIST_MAX_SIZE = None
```

### SOX2 or another marker looks too negative in phenotype maps

Try lowering that marker’s percentile:

```python
PHENOTYPE_POSITIVE_PERCENTILE_BY_CHANNEL = {
    1: 75,
    2: 60,
    3: 90,
}
```

or use validated manual/control thresholds instead of exploratory percentile thresholds.

### Phenotype maps look biologically wrong

Check in this order:

1. `debug/` channel QC images
2. `masks/` colony masks
3. `qc/` nuclei segmentation overlays
4. `cell_phenotypes/marker_intensity_threshold_diagnostics`
5. `cell_intensity_maps/`

Do not interpret phenotype classes until segmentation and thresholds pass QC.

---

## 22. Recommended default configuration

For most current PatternaV7 test runs:

```python
Z_MODE = "max"

MASK_MODE = "dapi"
DAPI_CHANNEL = 0
THRESH_METHOD = "percentile"
THRESH_PCT = 70

N_ANGLES = 240
N_R = 200
SIGMA_R = 2.5

SAVE_HEATMAPS = True
SAVE_DEBUG = True
SAVE_GRIDS = True
SAVE_MASKS = True

DO_CELLS = True
NUC_SEGMENTATION_MODE = "stardist"

STARDIST_PROB_THRESH = 0.30
STARDIST_NMS_THRESH = 0.55
STARDIST_MIN_SIZE = None
STARDIST_MAX_SIZE = None

EXPORT_CELL_PHENOTYPES = True
PHENOTYPE_THRESHOLD_MODE = "percentile_global"
PHENOTYPE_INTENSITY_STAT = "mean"
```

These defaults are good for testing and exploratory analysis, but final biological interpretation should use validated segmentation settings and calibrated phenotype thresholds.
