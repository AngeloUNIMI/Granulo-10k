import timm
import torch

def create_timm_encoder(model_name, pretrained=True):
    """
    Loads TIMM foundation models with classifier removed
    so the model outputs feature embeddings directly.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,   # removes classifier head
    )
    return model

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_last_layers(model, layer_substrings):
    # Re-enable gradient updates only for selected layers
    for name, param in model.named_parameters():
        if any(substr in name for substr in layer_substrings):
            param.requires_grad = True
            
def get_optimizer(
    model_name,
    modelEnsemble,
    phase="pretrain",
    base_lr=1e-3,
    criterion=None
):
    """
    Returns an optimizer for classification or regression, 
    with correct LR scaling and proper weight decay settings
    for CNNs and foundation models.
    """

    # Collect params
    paramsA = [p for p in modelEnsemble["encA"].parameters() if p.requires_grad]
    paramsB = [p for p in modelEnsemble["encB"].parameters() if p.requires_grad]
    paramsC = [p for p in modelEnsemble["encC"].parameters() if p.requires_grad]
    paramsDec = [p for p in modelEnsemble["dec"].parameters() if p.requires_grad]
    paramsUnc = [p for p in criterion.parameters() if criterion is not None] if criterion is not None else []

    # Determine model type
    cnn_backbones = [
        "resnet18","resnet34","resnet50","resnet101","resnet152",
        "resnext50_32x4d","resnext101_32x8d","resnext101_64x4d",
        "wide_resnet50_2","wide_resnet101_2"
    ]
    is_cnn = model_name in cnn_backbones
    is_foundation = any(k in model_name for k in ["vit","clip","eva","convnext"])

    # ======================================================================
    # PRETRAINING (CLASSIFICATION)
    # ======================================================================
    if phase == "classification":
        print(f"[Optimizer] PRETRAINING phase for {model_name}")

        # ---------------------------------------------------------------
        # CNN BACKBONES → use SGD, standard weight decay = 5e-4
        # ---------------------------------------------------------------
        if is_cnn:
            return torch.optim.SGD(
                [
                    {"params": paramsA,   "lr": base_lr},
                    {"params": paramsB,   "lr": base_lr},
                    {"params": paramsC,   "lr": base_lr},
                    {"params": paramsDec, "lr": base_lr},
                ],
                momentum=0.9,
                weight_decay=5e-4
            )

        # ---------------------------------------------------------------
        # FOUNDATION MODELS → use AdamW, WD=0.01 (standard)
        # ---------------------------------------------------------------
        elif is_foundation:
            return torch.optim.AdamW(
                [
                    {"params": paramsA,   "lr": base_lr * 0.5},  # safer LR
                    {"params": paramsB,   "lr": base_lr * 0.5},
                    {"params": paramsC,   "lr": base_lr * 0.5},
                    {"params": paramsDec, "lr": base_lr},
                ],
                weight_decay=0.01
            )

        else:
            raise ValueError(f"Unknown model type: {model_name}")

    # ======================================================================
    # REGRESSION PHASE (YOUR MAIN USE CASE)
    # ======================================================================
    elif phase == "regression":
        print(f"[Optimizer] REGRESSION phase for {model_name}")

        # ---------------------------------------------------------------
        # Learning rate rules for REGRESSION:
        # CNN encoder LR = 10% of base_lr
        # Foundation encoder LR = 5% of base_lr
        # Decoder LR = base_lr
        # Uncertainty LR = base_lr
        # ---------------------------------------------------------------
        if is_cnn:
            lr_enc = base_lr * 0.10      # moderate fine-tuning
            weight_decay_enc = 1e-4      # good for CNN regression
        elif is_foundation:
            lr_enc = base_lr * 0.05      # foundation models need tiny LR
            weight_decay_enc = 0.05      # standard WD for ViT/CLIP/ConvNeXt
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        lr_dec = base_lr
        lr_unc = base_lr

        # NO WEIGHT DECAY for decoder + uncertainty params
        weight_decay_dec = 0.0
        weight_decay_unc = 0.0

        # ---------------------------------------------------------------
        # CNN REGRESSION -> SGD
        # ---------------------------------------------------------------
        if is_cnn:
            return torch.optim.SGD(
                [
                    {"params": paramsA,   "lr": lr_enc, "weight_decay": weight_decay_enc},
                    {"params": paramsB,   "lr": lr_enc, "weight_decay": weight_decay_enc},
                    {"params": paramsC,   "lr": lr_enc, "weight_decay": weight_decay_enc},
                    {"params": paramsDec, "lr": lr_dec, "weight_decay": weight_decay_dec},
                    {"params": paramsUnc, "lr": lr_unc, "weight_decay": weight_decay_unc},
                ],
                momentum=0.9
            )

        # ---------------------------------------------------------------
        # FOUNDATION MODELS REGRESSION -> AdamW
        # ---------------------------------------------------------------
        elif is_foundation:
            return torch.optim.AdamW(
                [
                    {"params": paramsA,   "lr": lr_enc, "weight_decay": weight_decay_enc},
                    {"params": paramsB,   "lr": lr_enc, "weight_decay": weight_decay_enc},
                    {"params": paramsC,   "lr": lr_enc, "weight_decay": weight_decay_enc},
                    {"params": paramsDec, "lr": lr_dec, "weight_decay": weight_decay_dec},
                    {"params": paramsUnc, "lr": lr_unc, "weight_decay": weight_decay_unc},
                ]
            )

    else:
        raise ValueError(f"Invalid phase: {phase}")


def get_input_size(model_name):
    
    # -----------------------------
    # 1. CNN-BASED MODELS
    # -----------------------------
    cnn_models = [
        "resnet18","resnet34","resnet50","resnet101","resnet152",
        "resnext50_32x4d","resnext101_32x8d","resnext101_64x4d",
        "wide_resnet50_2","wide_resnet101_2",
    ]
    if model_name in cnn_models:
        return (320, 240)

    dino_models = ["dino_vitb14"] 
    if model_name in dino_models:
        return (518, 518)

    clip_models = ["clip_vitl14"]
    if model_name in clip_models:
        return (224, 224)

    eva_models = ["eva02_clip_l14"]
    if model_name in eva_models:
        return (336, 336)

    convnext_models = ["convnextv2_base"]
    if model_name in convnext_models:
        return (384, 384)

    print(f"[WARNING] Unknown model '{model_name}', using 224×224 fallback.")
    return (224, 224)