import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img

# def estimate_airlight_dc(dark_channel, image, percentile=90):
#     """Estimate airlight using the dark channel prior and the percentile of brightness."""
#     num_pixels = dark_channel.size
#     num_brightest = int(max(num_pixels * percentile / 100, 1))
#     dark_vec = dark_channel.ravel()
#     image_vec = image.reshape(num_pixels, 3)
#     indices = np.argsort(dark_vec)
#     indices = indices[-num_brightest:]
#     brightest = image_vec[indices]
#     max_intensity = brightest.max(axis=0)
#     return max_intensity

# def estimate_transmission(dark_channel, airlight, omega=0.95):
#     """Estimate transmission map using the airlight and omega."""
#     airlight_reshaped = airlight.reshape(1, 1, 3)
#     transmission = 1 - omega * dark_channel[:, :, None] / airlight_reshaped
#     transmission = np.clip(transmission, 0, 1)
#     return transmission

# def recover_scene(image, transmission, airlight, t0=0.1):
#     """Recover the scene radiance from the hazy image, transmission map, and airlight."""
#     refined_transmission = np.clip(transmission, a_min=t0, a_max=1)
#     image_recovered = (image - airlight) / refined_transmission + airlight
#     image_recovered = np.clip(image_recovered, a_min=0, a_max=255)
#     return image_recovered.astype(image.dtype)  # Preserve original data type

# def preprocess_hazy_image(image):
#     """Preprocess hazy image to reduce haze."""
#     # print(image.shape)
#     original_dtype = image.dtype  # Save original data type
#     image_float = image.astype(np.float32) / 255.0
#     dark_channel = cv2.min(cv2.min(image_float[:, :, 0], image_float[:, :, 1]), image_float[:, :, 2])
#     kernel_size = 15
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
#     dark_channel = cv2.erode(dark_channel, kernel)
#     airlight = estimate_airlight_dc(dark_channel, image_float)
#     transmission = estimate_transmission(dark_channel, airlight)
#     recovered_image = recover_scene(image_float, transmission, airlight)
#     recovered_image = (recovered_image * 255).astype(original_dtype)  # Convert back to original data type
#     # print(recovered_image.shape)
#     return recovered_image

def preprocess_hazy_image(image):
    # Scale the float32 image to the range [0, 255] and convert to uint8
    image_8bit = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Convert the 8-bit image to the LAB color space
    lab = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2Lab)
    # Split the LAB image to L, A, and B channels
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # Merge the CLAHE enhanced L channel with the original A and B channels
    limg = cv2.merge((cl, a, b))
    # Convert the LAB image back to BGR
    processed_img_8bit = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    # Convert the 8-bit processed image back to float32
    processed_img = processed_img_8bit.astype('float32') / 255.0
    
    return processed_img



def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
		# source_img = preprocess_hazy_image(source_img)
		
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}
