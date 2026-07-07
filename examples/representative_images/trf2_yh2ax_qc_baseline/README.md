# TRF2/yH2AX representative QC baseline assets

These derived assets are for Jules/Codex segmentation optimisation. They are not raw CZI files.

## Purpose

Use these examples to optimise DAPI nuclear segmentation, FITC/TRF2 spot segmentation, RhReX/gamma-H2AX spot segmentation, and 2D per-plane behaviour versus 3D all-Z behaviour.

The optimisation target is biology-constrained: do not make the images look cleaner by suppressing expected TRF2/telomere signal.

## Channel mapping

- `C0_RhReX_gH2AX`: RhReX / gamma-H2AX candidate channel.
- `C1_FITC_TRF2`: FITC / TRF2 telomere candidate channel.
- `C2_DAPI`: DAPI nuclear channel.

## Representative samples

- `Well1.czi`: WT palbociclib, typical 20x field.
- `Well2.czi`: WT palbociclib, second 20x baseline field.
- `Well4.czi`: 20x oversegmentation/failure case.
- `Well5.czi`: BRCA1-/- palbociclib, likely higher gamma-H2AX.
- `Well6.czi`: BRCA1-/- palbociclib, likely higher gamma-H2AX.
- `Well6 63x.czi`: separate 63x parameter-test case.

## Files

- `counts_2d_vs_3d_representative.csv`: per-image 2D and 3D first-pass counts.
- `derived_qc/*/qc_contact_sheet.pdf`: compact composite QC contact sheets.
- `derived_qc/*/qc_channels_contact_sheet.pdf`: compact single-channel QC contact sheets.
- `derived_qc/*/single_channel_pngs/`: labelled single-channel segmentation PNGs.
- `derived_qc/*/composite_pngs/`: labelled composite QC PNGs.
- `optimisation_crops/*/z*.png`: per-Z 2D crop slices.
- `optimisation_crops/*/stack.tif`: compact 8-bit display-scaled 3D crop stack.
- `optimisation_crops/*/metadata.json`: crop metadata without local absolute paths.

## Baseline standard for Jules

Treat 2D QC/contact sheets as the minimum baseline standard. The optimised 3D pipeline should not perform worse than the existing 2D scoring/segmentation standard unless the difference is explicitly justified as removal of clear background or artefact.

Important caveat: the current repo 2D mode uses the first Z plane for z-stacks. If an older validated 2D workflow used a different plane or projection, use the older validated counts as the stronger scoring guardrail. `Well2.czi` has very poor current first-Z 2D nuclear detection, so it should be treated as a warning case rather than a good nuclear baseline.

## Biology-constrained optimisation rules

- Preserve TRF2/telomere signal in WT palbociclib Wells 1 and 2.
- Do not optimise by simply minimising spot counts.
- Do not force BRCA1-/- Wells 5 and 6 to resemble WT fields.
- Do not tune gamma-H2AX downward only because BRCA1-/- has more signal.
- Produce separate 20x and 63x parameter presets.
- Stop after two optimisation rounds unless human review approves more.

## Safety notes

- No raw `.czi` files are included.
- Metadata contains source file names only, not local absolute paths.
- PNG/TIFF crop assets are 8-bit display-scaled derivatives for optimisation, not raw microscopy data.
