import torch
from torch import nn
from typing import Any, Optional, Tuple, Union
from transformers import CLIPVisionModelWithProjection, AutoProcessor
from PIL import Image





# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class ViTForSketchMatching(nn.Module):
    def __init__(self, model_name_or_path="openai/clip-vit-base-patch32"):
        super().__init__()
        
        self.screen_model = CLIPVisionModelWithProjection.from_pretrained(model_name_or_path)
        self.sketch_model = CLIPVisionModelWithProjection.from_pretrained(model_name_or_path)
        
        for param in self.screen_model.parameters():
            param.requires_grad = False
        
        logit_scale_init_value = 2.6592
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))


    def forward(
        self,
        screen_pixel_values: Optional[torch.FloatTensor] = None,
        sketch_pixel_values: Optional[torch.FloatTensor] = None,
    ):

        screen_embeds = self.screen_model(
            pixel_values=screen_pixel_values,
        ).image_embeds        
        sketch_embeds = self.sketch_model(
            pixel_values=sketch_pixel_values,
        ).image_embeds

        # normalized features
        screen_embeds = screen_embeds / screen_embeds.norm(p=2, dim=-1, keepdim=True)
        sketch_embeds = sketch_embeds / sketch_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(screen_embeds, sketch_embeds.t()) * logit_scale

        loss = clip_loss(logits_per_image)

        return {
            "loss": loss,
            "scree_embeds": screen_embeds,
            "sketch_embeds": sketch_embeds
        }

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = ViTForSketchMatching().to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image = Image.open("../test_images/_2MXoNnODJIXY33Yn_ShINh_ACuRKpdFisE6AM5w19kHiPnLWPbZpwsWYirYVm2siYqe.png")
    image_inputs = processor(images=[image], return_tensors="pt").to(device)
    
    output = model(image_inputs['pixel_values'], image_inputs['pixel_values'])
