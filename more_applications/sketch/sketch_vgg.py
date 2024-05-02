import torch
import torch.nn as nn
import torchvision.transforms as transforms

from random import choice
from typing import Any, Optional, Tuple, Union
from torchvision.models import vgg11
from PIL import Image
from transformers import AutoProcessor

def swire_loss(screen_embeds, sketch_embeds):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    epoch_size = len(screen_embeds)
    total_loss = torch.tensor(0.0, device=device)
    
    if epoch_size == 1: 
        return total_loss
    
    for n in range(epoch_size):
        # Get a random screen that is not pair of sketch n
        r = choice([i for i in range(0, epoch_size) if i != n])
        
        positive = torch.dist(sketch_embeds[n], screen_embeds[n])
        negative = torch.dist(sketch_embeds[n], screen_embeds[r])
    
        loss = positive + torch.max(torch.tensor(0.0, device=device), torch.tensor(0.2, device=device)-negative)
        total_loss += loss
    
    return total_loss

class VggForSketchMatching(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.screen_model = vgg11(num_classes=64)
        self.sketch_model = vgg11(num_classes=64)

    def forward(
        self,
        screen_pixel_values: Optional[torch.FloatTensor] = None,
        sketch_pixel_values: Optional[torch.FloatTensor] = None,
    ):

        screen_embeds = self.screen_model(screen_pixel_values)
        sketch_embeds = self.sketch_model(sketch_pixel_values)

        loss = swire_loss(screen_embeds, sketch_embeds)

        return {
            "loss": loss,
            "scree_embeds": screen_embeds,
            "sketch_embeds": sketch_embeds
        }


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = VggForSketchMatching().to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image = Image.open("../test_images/_2MXoNnODJIXY33Yn_ShINh_ACuRKpdFisE6AM5w19kHiPnLWPbZpwsWYirYVm2siYqe.png")    
    image_inputs = processor(images=[image], return_tensors="pt").to(device)

    output = model(image_inputs['pixel_values'], image_inputs['pixel_values'])
    
    print(output)