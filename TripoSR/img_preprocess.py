import numpy as np
from PIL import Image
from tsr.utils import remove_background, resize_foreground, save_video
import rembg
import os
from tsr.system import TSR

model = TSR.from_pretrained(
    "./model",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
device = "cpu"
model.to(device)

image = []
images = []
img_paths = []
for x in os.listdir("Data/a/2D"):
        if x.split('.')[2]=="png":
                img_paths.append(x)
output_dir = "exm/images"

for i, img_path in enumerate(img_paths):
        image = remove_background(Image.open("Data/a/2D/"+img_path), rembg.new_session())
        image = resize_foreground(image, 0.85)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        if not os.path.exists(os.path.join(output_dir, str(i))):
            os.makedirs(os.path.join(output_dir, str(i)))
        image.save(os.path.join(output_dir, str(i), f"input.png"))
        images.append(image)

scene_codes = model([images[0]], device = "cpu")
render_images = model.render(scene_codes, n_views=11, return_type="pil")
for ri, render_image in enumerate(render_images[0]):
        render_image.save(os.path.join(output_dir, str(ri), f"render_{ri:03d}.png"))