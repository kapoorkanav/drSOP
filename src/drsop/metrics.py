from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def compute_multilabel_metrics(y_true, y_prob, codes: list, threshold: float = 0.5) -> dict:
    """Per-disease AUC / F1 / sensitivity / specificity, plus their means across diseases.

    AUC is the headline number: it doesn't depend on a threshold, so it measures how well the
    model ranks sick eyes above healthy ones even when the 0.5 cut-off is badly placed for a
    rare disease. Threshold-based metrics are reported at `threshold` for context. AUC is NaN
    for a disease with only one class present in the split, and is skipped in the mean."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    aucs, f1s = [], []
    for i, code in enumerate(codes):
        t, p, s = y_true[:, i], y_pred[:, i], y_prob[:, i]
        auc = roc_auc_score(t, s) if len(np.unique(t)) == 2 else float("nan")
        tp = int(((p == 1) & (t == 1)).sum()); fn = int(((p == 0) & (t == 1)).sum())
        tn = int(((p == 0) & (t == 0)).sum()); fp = int(((p == 1) & (t == 0)).sum())
        out[f"{code}_auc"] = auc
        out[f"{code}_f1"] = f1_score(t, p, zero_division=0)
        out[f"{code}_sens"] = tp / (tp + fn) if (tp + fn) else float("nan")
        out[f"{code}_spec"] = tn / (tn + fp) if (tn + fp) else float("nan")
        if not np.isnan(auc):
            aucs.append(auc)
        f1s.append(out[f"{code}_f1"])
    out["mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["mean_f1"] = float(np.mean(f1s))
    return out
