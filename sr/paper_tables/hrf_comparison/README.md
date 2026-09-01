# HRF comparison table

This directory contains a standalone IEEE-style HRF comparison table.

## Files

- `hrf_table.tex`: minimal compilable document.
- `hrf_table_content.tex`: table environment that can be copied into a paper.
- `new_references.bib`: BibTeX entries for the newly introduced LoFTR and R2D2 references.

## Build

From this directory:

```powershell
latexmk -pdf -interaction=nonstopmode -file-line-error hrf_table.tex
```

The literal reference numbers follow the current ICME manuscript: existing
references retain their original numbers, with LoFTR and R2D2 provisionally
appended as `[24]` and `[25]`. In a full manuscript, replace literal numbers
with `\cite{}` commands so LaTeX assigns them automatically.

The standalone wrapper sets the table counter so the preview is labeled
`TABLE II`, matching the current manuscript. Remove that counter adjustment
when copying only `hrf_table_content.tex` into a complete paper.

The table-level SalC values use all 18 HRF pairs. Missing LoFTR and SuperPoint
aligned outputs are evaluated as identity registrations by retaining the source
image. Added-method times are historical and were not measured on unified
hardware.
