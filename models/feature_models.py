import torch
import torch.nn as nn
import timm

def get_feature_extractor(model_name, device, weights_dir=None):
    """
    Dynamically loads the requested neural network architectures.
    
    Args:
        model_name (str): The configuration string (e.g. 'rn50', 'uni', 'hibou_b')
        device (torch.device): CPU or CUDA device to cast the model to.
        weights_dir (str, optional): The path resolving to local downloaded weights if HF Hub is not used.
        
    Returns:
        model (nn.Module): The loaded, evaluation-ready PyTorch architecture.
    """
    model_name = model_name.lower().strip()
    
    # Baseline CNNs
    if model_name == 'rn18':
        import torchvision
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Identity() # Remove classification head
        
    elif model_name == 'rn50':
        import torchvision
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        
    elif model_name == 'densenet121':
        import torchvision
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Identity()
        
    # Standard Vision Transformers
    elif model_name == 'vit_l':
        model = timm.create_model("vit_large_patch16_224", num_classes=0, pretrained=True)
        
    # Pathology Foundation Models
    elif model_name == 'uni':
        # UNI requires access to huggingface or local weights
        model = timm.create_model(
            "vit_large_patch16_224", 
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True
        )
        if weights_dir:
            import os
            weights_path = os.path.join(weights_dir, "vit_large_patch16_224.dinov2.uni_mass100k", "pytorch_model.bin")
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        else:
            # Fallback to direct HF Hub lookup if authorized
            model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)

    elif model_name == 'provgigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        
    elif model_name == 'phikon':
        from transformers import ViTModel
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        
    elif model_name == 'virchow':
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU)
        
    elif model_name == 'virchow2cls':
        model = timm.create_model("hf-hub:paige-ai/Virchow-2", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU)
        
    elif model_name == 'hibou_l':
        from transformers import AutoModel
        model = AutoModel.from_pretrained("histai/hibou-l", trust_remote_code=True)
        
    elif model_name == 'optimus':
        model = timm.create_model("hf_hub:bioptimus/H-optimus-0", pretrained=True)
        
    else:
        raise ValueError(f"Feature Extractor '{model_name}' is not currently supported or recognized.")
        
    # Prepare model for inference
    model = model.to(device)
    model.eval()
    
    # Explicitly calculate and print the parameter size for verification
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_mb = (param_size + buffer_size) / (1024**2)
    print(f"Loaded {model_name} onto {device} | Size: {size_mb:.2f} MB")
    
    return model


def forward_features(model, batch, model_name):
    """
    Standardizes the output from various architectures.
    Different Foundation Models return different objects (e.g. HuggingFace outputs vs TIMM tensors).
    This extracts the final [N, D] feature tensor regardless of architecture nuances.
    """
    model_name = model_name.lower().strip()
    
    with torch.no_grad():
        features = model(batch)
        
        if model_name == 'phikon':
            features = features.last_hidden_state[:, 0, :]
            
        elif model_name == 'virchow':
            class_token = features[:, 0]
            patch_tokens = features[:, 1:]
            features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            
        elif model_name == 'virchow2cls':
            # Rely strictly on the CLS token
            features = features[:, 0]
            
        elif model_name == 'hibou_l' or model_name == 'hibou_b':
            features = features.pooler_output
            
        return features.cpu()
