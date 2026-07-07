# CZI Nuclear Foci Analysis

Config-driven local analysis for 3-channel ZEISS CZI images with:

- nuclear segmentation from one channel;
- independent segmentation of two nuclear DNA-damage focus channels;
- co-localisation calls between the two focus channels inside nuclei;
- mask TIFFs and channel-specific QC PNGs for manual review;
- per-image, per-nucleus, per-focus and co-localised-pair CSV outputs;
- sensitivity, specificity and selectivity tables for simple per-nucleus count cut-offs.

The package is designed for local research workflows. It does not modify raw CZI
files and does not upload image data.

## Install

From this directory:

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The package tests use the standard library `unittest`, so no separate test
runner is required.

## Run A New 3-Channel Experiment

1. Copy and edit the example config:

```zsh
cp configs/three_channel_dna_damage.example.json configs/my_experiment.json
cp configs/example_condition_map.csv configs/my_condition_map.csv
```

2. Set the channel definitions in the config:

- `nuclei.channel_index`: nuclear marker, usually DAPI.
- `focus_a.channel_index`: first focus channel.
- `focus_b.channel_index`: second focus channel.
- `focus_a.name` and `focus_b.name`: short output-safe names, for example
  `AF488`, `gH2AX`, `53BP1` or `RAD51`.
- `colocalization_dilation_px`: pixel tolerance for overlap. A value of `1`
  means a focus is co-localised if it intersects the other focus after 1-pixel
  dilation.
- `control_conditions`: conditions treated as negative controls for threshold
  sensitivity/specificity calculations.
- `condition_map_csv`: optional CSV mapping filename codes to biological
  condition names. It must contain `condition_code` and `condition` columns.
  Use a path relative to the config file.

3. Run a one-file 2D QC pass:

```zsh
czi-foci analyse \
  --root "/path/to/czi_folder" \
  --config configs/my_experiment.json \
  --output-dir output/my_experiment_2d_qc \
  --mode 2d \
  --limit 1 \
  --qc-title "My experiment 2D QC"
```

4. Inspect the output masks and QC overlays. If segmentation is acceptable,
run the full folder:

```zsh
czi-foci analyse \
  --root "/path/to/czi_folder" \
  --config configs/my_experiment.json \
  --output-dir output/my_experiment_2d \
  --mode 2d \
  --qc-title "My experiment 2D QC"
```

Use `--overwrite` only when you intentionally want to replace an existing output
directory.


## 3D Z-Stack Analysis

Use 3D mode for CZI z-stacks when foci should be counted and colocalised in
physical 3D space rather than on a single 2D plane. In 3D mode the pipeline
expects voxel spacing in the CZI metadata and computes colocalisation using
micron-scaled centroid distances.

### 3D config keys

Add or edit these top-level JSON keys in the experiment config:

```json
{
  "mode": "3d",
  "colocalization_distance_um": 1.0,
  "nucleus_strategy": "mip_2d_propagate",
  "z_projection": "first"
}
```

- `mode`: `2d` or `3d`. The CLI `--mode` argument overrides the config.
- `colocalization_distance_um`: maximum physical distance between 3D focus
  centroids for a colocalisation call.
- `nucleus_strategy`: `mip_2d_propagate` is the recommended default. It creates
  a 2D nuclear mask from a maximum-intensity projection and propagates it across
  Z. `volume_3d` is available for experimental fully volumetric nuclear
  thresholding.
- `z_projection`: currently only `first` is implemented for 2D mode. This
  preserves the historical behaviour of using the first Z plane when a z-stack
  is analysed as 2D.

### One-file 3D QC run

Start with one representative file before running a whole folder:

```zsh
czi-foci analyse \
  --root "/path/to/czi_folder" \
  --config configs/my_experiment.json \
  --output-dir output/my_experiment_3d_qc \
  --mode 3d \
  --limit 1 \
  --qc-title "My experiment 3D QC"
```

### Full 3D folder run

```zsh
czi-foci analyse \
  --root "/path/to/czi_folder" \
  --config configs/my_experiment.json \
  --output-dir output/my_experiment_3d \
  --mode 3d \
  --qc-title "My experiment 3D QC"
```

### Manifest-based run

Use a manifest when you want exact control over which CZI files are analysed.
The CSV must contain a `source_path` column.

```csv
source_path
/path/to/file_001.czi
/path/to/file_002.czi
```

```zsh
czi-foci analyse \
  --manifest manifests/my_subset.csv \
  --config configs/my_experiment.json \
  --output-dir output/my_subset_3d \
  --mode 3d \
  --qc-title "My subset 3D QC"
```

> **Note on 2D z-stack handling:** When processing a z-stack file in `2d` mode, the pipeline will default to using the first (lowest index) Z-plane (`z_projection = "first"`). You can explicitly configure this via the `z_projection` parameter in the JSON config (currently only `first` is fully implemented; `min`/`max_mip` may be added in the future).

> **Note on 3D segmentation parameters:** The default parameters for `min_area_px` and `tophat_radius_px` may not transfer across magnifications or acquisition settings. Higher magnification z-stacks often require different `tophat_radius_px`, `min_area_px`, and threshold multipliers. Tune parameters on QC overlays before treating counts as biological measurements.

> **Note on missing Z metadata:** 3D mode requires physical Z spacing in the CZI metadata. Files without Z spacing should be analysed in 2D mode or excluded from a 3D run.

## Output Files


Each run writes:

- `resolved_config.json`: exact config used for the run.
- `image_summary.csv`: field-level counts and CZI channel metadata.
- `nucleus_measurements.csv`: nucleus area or volume, intensities and
  per-nucleus focus counts.
- `<focus_a.name>_measurements.csv`: per-focus measurements for focus A.
- `<focus_b.name>_measurements.csv`: per-focus measurements for focus B.
- `colocalized_focus_pairs.csv`: focus-pair calls.
- `condition_summary.csv`: condition/timepoint summary.
- `threshold_performance_<metric>.csv`: sensitivity, specificity and selectivity
  across candidate thresholds.
- `best_thresholds.csv`: best threshold by Youden J, then specificity, then
  sensitivity.
- `masks/*.tif`: nuclei, focus A, focus B and co-localised focus masks.
- `qc_overlays/*.png`: composite QC overlays.
- `qc_channels/*.png`: single-channel QC views.
- `qc_contact_sheet.pdf`: review PDF of composite overlays.

In 3D mode, measurement CSVs include Z/Y/X centroid columns and voxel-size
columns:

- `voxel_size_z_um`, `voxel_size_y_um`, `voxel_size_x_um`
- `nucleus_volume_voxels`, `nucleus_volume_um3`
- `focus_volume_voxels`, `focus_volume_um3`
- `nucleus_centroid_z_px`, `nucleus_centroid_y_px`, `nucleus_centroid_x_px`
- `focus_centroid_z_px`, `focus_centroid_y_px`, `focus_centroid_x_px`

## Developer and Bioinformatics Handoff

Recommended handoff files for downstream statistics:

1. `resolved_config.json` for exact parameters.
2. `image_summary.csv` for one row per image or field.
3. `nucleus_measurements.csv` for per-nucleus modelling.
4. `<focus_a.name>_measurements.csv` and `<focus_b.name>_measurements.csv` for
   per-focus distributions.
5. `colocalized_focus_pairs.csv` for pair-level colocalisation checks.
6. `qc_contact_sheet.pdf` and `qc_overlays/*.png` for manual QC sign-off.

Suggested developer checks before handing off results:

```zsh
python3 -m py_compile src/czi_foci/*.py
PYTHONPATH=src python3 -m unittest discover -s tests
```

For reproducible project runs, keep these under version control:

- experiment config JSON files in `configs/`;
- non-sensitive manifest CSVs in `manifests/`;
- analysis scripts or notebooks that consume the output CSVs.

Do not commit raw CZI files, local absolute paths, credentials, or temporary
analysis outputs.

## Current Limitations

- The bundled parser supports uncompressed Gray8 and Gray16 CZI planes. If a CZI
  uses JPEG XR, mosaics, pyramids or other compressed variants, use a validated
  CZI reader before analysis.
- Threshold performance treats non-control conditions as positive examples. This
  is suitable for screening-level separation from background, not for proving
  biological mechanism.
- Cross-experiment comparisons remain vulnerable to staining, acquisition and
  batch effects unless normalised to a confirmed control strategy.

## Verify

```zsh
python3 -m py_compile src/czi_foci/*.py
PYTHONPATH=src python3 -m unittest discover -s tests
```
