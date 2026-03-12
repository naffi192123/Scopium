import torch
from torchvision import transforms

def get_transform(transform_name):
    """
    Returns a torchvision transform or custom normalization class 
    based on the provided string key.
    """
    transform_name = transform_name.lower().strip()
    
    if transform_name == 'imagenet':
        return transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
    elif transform_name == 'resnet50lunit_default':
        return transforms.Normalize(mean=(0.70322989, 0.53606487, 0.66096631), std=(0.21716536, 0.26081574, 0.20723464))
        
    elif transform_name == 'hibou_default':
        return transforms.Normalize(mean=[0.7068, 0.5755, 0.7220], std=[0.1950, 0.2316, 0.1816])
        
    elif transform_name == 'optimus_default':
         return transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517))
         
    elif transform_name == 'histo_resnet18':
         return transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
         
    elif transform_name == 'kaiko_default':
         return transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
         
    elif transform_name == 'resize_224':
        return transforms.Resize(224)
        
    elif transform_name == 'resize_256':
        return transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC)
        
    elif transform_name == 'center_crop_224':
        return transforms.CenterCrop(224)
        
    elif transform_name == 'totensor':
        return transforms.ToTensor()
        
    elif transform_name == 'reinhard':
        try:
            import torchstain
        except ImportError:
            raise ImportError("Please install torchstain via `pip install torchstain` to use Reinhard normalization.")
            
        class ReinhardNormalisation:
            def __init__(self):
                self.normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
                # Optional: specific targets from Ovarian_Features if required.
                # self.normalizer.target_means = torch.tensor([79.2929, 11.2809, -5.9533])
                # self.normalizer.target_stds = torch.tensor([17.3957,  8.6891, 10.5019])

            def __call__(self, image):
                # torchstain expects (H, W, C) for Reinhard mapping natively
                # If image is a tensor (C, H, W), permute it.
                if isinstance(image, torch.Tensor):
                    img_hwc = image.permute(1, 2, 0)
                else:
                    img_hwc = transforms.ToTensor()(image).permute(1, 2, 0)
                
                # Scale up to 0-255 if tensor is 0-1
                if img_hwc.max() <= 1.0:
                    img_hwc = img_hwc * 255.0
                    
                norm = self.normalizer.normalize(I=img_hwc)
                # Return back to (C, H, W) normalized to [0,1]
                norm = norm.permute(2, 0, 1) / 255.0
                return norm
                
        return ReinhardNormalisation()
        
    elif transform_name == 'macenko':
        try:
            import torchstain
        except ImportError:
            raise ImportError("Please install torchstain via `pip install torchstain` to use Macenko normalization.")
            
        class MacenkoNormalisation:
            def __init__(self):
                self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

            def __call__(self, image):
                if isinstance(image, torch.Tensor):
                    img_hwc = image.permute(1, 2, 0)
                else:
                    img_hwc = transforms.ToTensor()(image).permute(1, 2, 0)
                    
                if img_hwc.max() <= 1.0:
                    img_hwc = img_hwc * 255.0
                
                try:
                    norm, _, _ = self.normalizer.normalize(I=img_hwc, stains=False)
                    norm = norm.permute(2, 0, 1) / 255.0
                except:
                    # Fallback if Macenko geometry fails
                    norm = img_hwc.permute(2, 0, 1) / 255.0
                return norm
                
        return MacenkoNormalisation()
        
    elif transform_name == 'colorjitter':
        return transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        
    else:
        raise ValueError(f"Transform '{transform_name}' is not supported yet.")


def build_transform_pipeline(config):
    """
    Builds a torchvision Compose pipeline based on the yaml configuration.
    
    Example Config:
    feature_extraction:
      target_patch_size: 224
      transforms:
        - reinhard
        - imagenet
    """
    transform_list = []
    feat_config = config.get('feature_extraction', {})
    
    # 1. Resize if target_patch_size is specified
    target_size = feat_config.get('target_patch_size', -1)
    if target_size > 0:
        transform_list.append(transforms.Resize(target_size))
        
    # 2. Convert to Tensor (most custom normalizers and models expect generic PyTorch tensors)
    transform_list.append(transforms.ToTensor())
    
    # 3. Append dynamic transforms from the config list
    transforms_config = feat_config.get('transforms', [])
    for t_name in transforms_config:
        # Avoid duplicate to_tensor if user passed it
        if t_name.lower().strip() == 'totensor':
            continue
        transform_list.append(get_transform(t_name))
        
    return transforms.Compose(transform_list)
