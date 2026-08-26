from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
