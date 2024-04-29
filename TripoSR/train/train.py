import numpy as np
import rembg
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video

# Prepare the data



images=[]
# Process the image
rembg_session = rembg.new_session()
for i, image_path in enumerate(args.image):
        image = remove_background(Image.open(image_path), rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        images.append(image)

# 옵티마이저 및 손실 함수 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Load the model
model = TSR.from_pretrained(
    "./",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# Set parameters
chunkSize = 8192 # Chunk size
nViews = 30 # Number of views
mcResolution = 256 # Marching cubes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.renderer.set_chunk_size(chunkSize)

# Training loop
for epoch in range(100):  # number of epochs
    for image in images:
        # Forward pass
        with torch.no_grad():
            scene_codes = model([image], device=device)

        render_images = model.render(scene_codes, n_views=nViews, return_type="pil")
        meshes = model.extract_mesh(scene_codes, resolution=mcResolution)

        
        # Compute the loss
        """
        def compute_loss(self, render_out, render_gt):
            # NOTE: the rgb value range of OpenLRM is [0, 1]
            render_images = render_out['render_images']
            target_images = render_gt['target_images'].to(render_images)
            render_images = rearrange(render_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
            target_images = rearrange(target_images, 'b n ... -> (b n) ...') * 2.0 - 1.0

            loss_mse = F.mse_loss(render_images, target_images)
            loss_lpips = 2.0 * self.lpips(render_images, target_images)

            render_alphas = render_out['render_alphas']
            target_alphas = render_gt['target_alphas']
            loss_mask = F.mse_loss(render_alphas, target_alphas)

            loss = loss_mse + loss_lpips + loss_mask

            prefix = 'train'
            loss_dict = {}
            loss_dict.update({f'{prefix}/loss_mse': loss_mse})
            loss_dict.update({f'{prefix}/loss_lpips': loss_lpips})
            loss_dict.update({f'{prefix}/loss_mask': loss_mask})
            loss_dict.update({f'{prefix}/loss': loss})

            return loss, loss_dict
        """


        loss_mask = criterion(render_images, image)
        
        # Backward pass
        optimizer.zero_grad()
        loss_mask.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")