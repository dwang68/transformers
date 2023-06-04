import numpy, torch, os
print(numpy.__version__)
print(numpy.__file__)
print(os.environ.get('PYTHONPATH'))
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.__file__)

import requests
from PIL import Image
from IPython.display import display
from src.transformers.models.blip.modeling_blip import BlipForConditionalGeneration
from src.transformers.models.blip.processing_blip import BlipProcessor
# from transformers import BlipForConditionalGeneration
# from transformers import BlipProcessor
from loop.config import get_prompts

prompts = get_prompts()


class Captioner:
    def __init__(self):
        # Initialize the captioning model using the provided model path
        if os.path.exists("/data/scratch/projects/punim0478/dalinw"):
            tf_cache_dir = "/data/scratch/projects/punim0478/dalinw/.cache/huggingface/transformers"
        else:
            tf_cache_dir = "/home/dalinw/.cache/huggingface/transformers"
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=tf_cache_dir)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
                                                                  cache_dir=tf_cache_dir)

    # given an image, this generates a conditioned and a unconditioned caption
    def generate_captions(self, raw_image):
        # conditional image captioning
        text = "a photo of a mom"
        inputs = self.processor(raw_image, text, return_tensors="pt")
        out_tensor = self.model.generate(**inputs)
        conditioned_caption = self.processor.decode(out_tensor[0], skip_special_tokens=True)

        # unconditional image captioning
        inputs = self.processor(raw_image, return_tensors="pt")
        out_tensor = self.model.generate(**inputs, do_sample=True,  # turn on sampling
                                         max_length=100,  # maximum length of the generated text
                                         top_p=0.9)
        unconditioned_caption = self.processor.decode(out_tensor[0], skip_special_tokens=True)
        return conditioned_caption, unconditioned_caption

def show_prompt(captioner, index):
    for j in range(1):  # image sub variation index
        image = Image.open(os.path.join("images", f"image_{index}_{j}.png"))
        display(image)
        caption_tuple = captioner.generate_captions(image)
        print(
            f"Captions for instance {index}, sub-image index {j} with prompt '{prompts[index]}':\n'{caption_tuple[0]}'\n'{caption_tuple[1]}'")


captioner = Captioner()
show_prompt(captioner, 20)