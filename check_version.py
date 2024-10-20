from transformers import CLIPModel

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Iterate through all modules in the model and print the modules containing Attention
print("Attention modules in the CLIP model:")
for name, module in model.named_modules():
    if "Attention" in module.__class__.__name__:
        print(f"Module name: {name}, type: {module.__class__.__name__}")


from diffusers import UNet2DConditionModel

# Load the UNet model (UNet2DConditionModel from diffusers)
unet_model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# Iterate through all modules in the UNet model and print modules containing Attention and GEGLU
print("Attention and GEGLU modules in the UNet model:")

for name, module in unet_model.named_modules():
    if "Attention" in module.__class__.__name__ or "GEGLU" in module.__class__.__name__:
        print(f"Module name: {name}, type: {module.__class__.__name__}")
