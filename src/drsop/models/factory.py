from drsop.models.fusion_model import DRFusionModel
from drsop.models.image_only_model import ImageOnlyModel


def build_model(mcfg: dict, categorical_cardinalities: list, n_numeric: int,
                n_comorbidities: int, num_classes: int):
    """Picks the model from the config.

    `model.image_only: true` gives the no-metadata control (RETFound + LoRA -> head). Anything
    else -- including every config written before this flag existed -- gives the gated-fusion
    model exactly as before, so previous experiments are unaffected.
    """
    if mcfg.get("image_only", False):
        # proj_dim is optional here: absent or null means no projection, so the backbone's
        # native embedding goes straight into the head.
        return ImageOnlyModel(
            retfound_cfg=mcfg,
            head_cfg=mcfg["head"],
            proj_dim=mcfg.get("proj_dim"),
            num_classes=num_classes,
        )
    return DRFusionModel(
        retfound_cfg=mcfg,
        meta_cfg=mcfg["meta_encoder"],
        gate_cfg=mcfg["gate"],
        head_cfg=mcfg["head"],
        categorical_cardinalities=categorical_cardinalities,
        n_numeric=n_numeric,
        n_comorbidities=n_comorbidities,
        proj_dim=mcfg["proj_dim"],
        num_classes=num_classes,
    )
