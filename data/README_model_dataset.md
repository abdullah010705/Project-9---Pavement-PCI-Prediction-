# The modelling datasets

Built by `src/data/build_dataset.py` from `master_with_pci_3.csv`, in which PCI is
computed from distress only — rutting was removed from the PCI score after the client
review, because they are two different condition measurements.

| File | Target | Rows | Physical roads | Pavement lives |
|---|---|---|---|---|
| `model_dataset_pci.csv` | `PCI_NEXT` | 976 | 259 | 327 |
| `model_dataset_rut.csv` | `RUT_NEXT` | 693 | 207 | 257 |

`split_assignment.csv` holds the train/test decision for each physical road. It is
assigned **once** and applied to both datasets, so a road is either training in both or
testing in both. That is what prevents a road being learned from and then scored on.

## One row

One row is one road survey, paired with the condition recorded at that road's **next**
survey. Surveys with no following survey are dropped, because they have no answer to
predict.

- `PHYSICAL_ROAD_ID` — the physical stretch of road. The grouping key for the split.
- `ROAD_ID` — one pavement life. A rebuild starts a new one, so deterioration is never
  tracked across a resurfacing.

## Columns

Age, traffic loading, climate, pavement structure, maintenance history, current
condition, and lagged values from the previous survey (`PCI_PREV`, `RUT_PREV`,
`PCI_CHANGE_SINCE_LAST` and similar). The PCI table also carries `NEEDS_REPAIR`, which
is 1 where `PCI_NEXT` falls below 55 — the client's repair threshold.

Missing values are left as blanks rather than filled in here; each modelling stage
handles them, fitting any imputation on training rows only.

## Known limitations

- `DAYS_SINCE_LAST_PATCH` is missing for most rows, because most sections have no
  recorded patch. The missingness is meaningful and should not be imputed away.
- Lagged columns are empty for the first survey of each pavement life. Expected.
- Climate and traffic have gaps inherited from the LTPP source tables.
