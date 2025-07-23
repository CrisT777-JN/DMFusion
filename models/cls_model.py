import torch
import torch.nn as nn
import clip

 # Define custom LoRALinear class
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1):
        super(LoRALinear, self).__init__()
        self.r = r
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features)

        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(in_features, r) * (alpha / r))
            self.lora_B = nn.Parameter(torch.randn(r, out_features) * (alpha / r))
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        if self.r > 0:
            lora_out = torch.matmul(x, self.lora_A).matmul(self.lora_B)
            return self.linear(x) + lora_out
        else:
            return self.linear(x)

class LoraCLIP(nn.Module):
    def __init__(self, num_classes, clip_model='ViT-B/32', r=4, alpha=1, pretrained=True):
        super(LoraCLIP, self).__init__()

        # Load pretrained CLIP model
        self.clip_model, _ = clip.load(clip_model, device="cuda")
        self.clip_model.float()

        # Only unfreeze image encoder weights
        if pretrained:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Iterate through the visual module and replace linear layers with LoRA layers
        visual_modules = list(self.clip_model.visual.named_modules())
        for name, module in visual_modules:
            if isinstance(module, nn.Linear):
                in_features, out_features = module.in_features, module.out_features
                # Replace with LoRALinear
                setattr(self.clip_model.visual, name, LoRALinear(in_features, out_features, r, alpha))

        # Get the output dimension of the image encoder
        image_encoding_dim = self.clip_model.visual.output_dim

        # Define fully connected layers for the classifier
        self.fc1 = nn.Linear(image_encoding_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Extract features using CLIP's image encoder
        x = self.clip_model.encode_image(x)
        x = self.relu(self.fc1(x))
        feature = x
        x = self.fc2(x)
        return x, feature



