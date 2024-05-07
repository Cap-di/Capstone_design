import numpy as np
import rembg
import torch
import lpips
import os
from PIL import Image
import matplotlib.pyplot as plt

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

output_dir = "test/"

images=[]
rembg_session = rembg.new_session()
# for i, image_path in enumerate(args.image):
#         image = remove_background(Image.open(image_path), rembg_session)
#         image = resize_foreground(image, args.foreground_ratio)
#         image = np.array(image).astype(np.float32) / 255.0
#         image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
#         image = Image.fromarray((image * 255.0).astype(np.uint8))
#         images.append(image)


image = remove_background(Image.open("./examples/chair.png"), rembg_session)
image = resize_foreground(image, 0.85)
image = np.array(image).astype(np.float32) / 255.0
image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
image = Image.fromarray((image * 255.0).astype(np.uint8))
image.save(os.path.join(output_dir, f"input.png"))
images.append(image)

# Load the model
model = TSR.from_pretrained(
    "./train",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# Set parameters
chunkSize = 8192 # Chunk size
nViews = 30 # Number of views
mcResolution = 256 # Marching cubes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device=="cuda:0": torch.cuda.empty_cache()
model.to(device)
model.renderer.set_chunk_size(chunkSize)

for i, image in enumerate(images):
    with torch.no_grad():
        scene_codes = model([image], device=device)
        
    render_images = model.render(scene_codes, n_views=nViews, return_type="pil")
    for ri, render_image in enumerate(render_images[0]):
        render_image.save(os.path.join(output_dir, f"render_{ri:03d}.png"))