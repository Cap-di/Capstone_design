from tsr.utils import ImagePreprocessor, get_spherical_cameras

n_views = 384
elevation_deg = 330
camera_distance = 330
fovy_deg = 60.0  # 임의로 지정
height = 384
width = 288

# Get camera rays
rays_o, rays_d = get_spherical_cameras(
    n_views, elevation_deg, camera_distance, fovy_deg, height, width
)

print("Shape of rays_o: ", rays_o.shape)
print("Shape of rays_d: ", rays_d.shape)

print("value of rays_o: ", rays_o)
print("value of rays_d: ", rays_d)

