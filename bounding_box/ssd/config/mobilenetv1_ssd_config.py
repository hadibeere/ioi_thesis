import numpy as np

from ssd.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
mean = 583.6798
std = 70.0333
image_mean = np.array([mean, mean])
image_std = std
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2
weights = [0.04,1.0]
alpha = 1.0

specs = [
    SSDSpec(19, 16, SSDBoxSizes(20, 65), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(65, 110), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(110, 155), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(155, 200), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(200, 245), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(245, 290), [2, 3])
]
start_fm = 32
num_priors = [6,6,6,6,6,6]
channels_priors = [512, 1024, 512, 256, 256, 256]

priors = generate_ssd_priors(specs, image_size)
