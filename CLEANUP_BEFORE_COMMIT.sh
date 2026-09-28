#!/usr/bin/env bash
# Removes v1 files that v0.0.2 replaced. Run once from this folder, then commit.
# (Claude can create and overwrite files in this folder but cannot delete them,
#  so these removals have to be run locally.)
set -e
cd "$(dirname "$0")"

echo "Removing v1 files superseded by v0.0.2 ..."

# stray artefact
rm -f .__testdel

# v1 single-target module — replaced by src/models/compare_vs_baselines.py
git rm -q --ignore-unmatch src/cross_validate.py
rm -f src/cross_validate.py

# v1 results helper lived under src/; v0.0.2 has it at the repo root
git rm -q --ignore-unmatch src/results_io.py
rm -f src/results_io.py

# v1 single dataset — replaced by model_dataset_pci.csv + model_dataset_rut.csv
git rm -q --ignore-unmatch data/model_dataset.csv
rm -f data/model_dataset.csv

# v1 raw extract — v0.0.2 builds from master_with_pci_3.csv (rutting removed from PCI)
git rm -q --ignore-unmatch data/master_with_pci_2.csv
rm -f data/master_with_pci_2.csv

# v1 planning docs — superseded by INTEGRATION_NOTES.md and the v2 allocation doc
git rm -q --ignore-unmatch MERGE_NOTES.md ALLOCATION_STATUS.md
rm -f MERGE_NOTES.md ALLOCATION_STATUS.md

rm -rf __pycache__ src/__pycache__ src/models/__pycache__ src/data/__pycache__

echo "Done. Now:"
echo "  git add -A"
echo "  git status          # check before committing"
