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

3. Run a one-file QC pass:

```zsh
czi-foci analyse \
  --root "/Volumes/Backup/TS runs/Time Series V (24+48hr)" \
  --config configs/three_channel_dna_damage.example.json \
  --output-dir output/generic_qc_test \
  --limit 1 \
  --qc-title "Generic CZI Foci QC"
```

4. Inspect the output masks and QC overlays. If segmentation is acceptable,
run the full folder:

```zsh
czi-foci analyse \
  --root "/Volumes/Backup/TS runs/Time Series V (24+48hr)" \
  --config configs/three_channel_dna_damage.example.json \
  --output-dir output/generic_time_series_V \
  --qc-title "Time Series V Generic Nuclear Foci QC"
```

Use `--overwrite` only when you intentionally want to replace an existing output
directory.

## Output Files

Each run writes:

- `resolved_config.json`: exact config used for the run.
- `image_summary.csv`: field-level counts and CZI channel metadata.
- `nucleus_measurements.csv`: nucleus area, intensities and per-nucleus focus counts.
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
