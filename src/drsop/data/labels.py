def apply_label_map(labels, label_map: dict):
    """Collapses the ICDR grades into coarser classes, e.g. {0:0, 1:1, 2:1, 3:2, 4:2} for
    none / mid / severe. Applied once when a split is loaded, so everything downstream --
    class weights, confusion matrices, metrics -- sees the merged labels automatically.

    Returns labels unchanged when label_map is None, which is what every config that predates
    this does."""
    if not label_map:
        return labels
    mapped = labels.map(label_map)
    if mapped.isna().any():
        unmapped = sorted(labels[mapped.isna()].unique().tolist())
        raise ValueError(
            f"label_map {label_map} has no entry for grade(s) {unmapped} present in the data."
        )
    return mapped.astype(int)
