import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img

def preprocess_hazy_image(image):
    # Convert image to YCrCb color space
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb_image)

    # Process Y channel using spectral decomposition
    y_channel = channels[0].astype(np.float32) / 255.0
    dft = cv2.dft(y_channel, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Define the radius for the low frequency region
    rows, cols = y_channel.shape
    crow, ccol = rows // 2, cols // 2
    r = 30

    # Create a circular mask for low pass filtering - to keep low frequencies
    low_pass_mask = np.zeros((rows, cols, 2), np.uint8)
    center = (crow, ccol)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    low_pass_mask[mask_area] = 1

    # Apply the low pass mask to the DFT of the Y channel
    low_frequencies = dft_shift * low_pass_mask

    # Create a circular mask for high pass filtering - to remove low frequencies
    high_pass_mask = np.ones((rows, cols, 2), np.uint8)
    high_pass_mask[mask_area] = 0

    # Apply the high pass mask to the DFT of the Y channel
    high_frequencies = dft_shift * high_pass_mask

    # Combine the low and high frequencies
    combined_frequencies = low_frequencies * 0.1 + high_frequencies * 1.5

    # Inverse DFT to convert back to the spatial domain
    combined_ishift = np.fft.ifftshift(combined_frequencies)
    combined_back = cv2.idft(combined_ishift)
    combined_back = cv2.magnitude(combined_back[:, :, 0], combined_back[:, :, 1])

    # Normalize the combined image to bring the values between 0 and 1
    combined_back = cv2.normalize(combined_back, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Convert back to 8-bit
    combined_back = cv2.convertScaleAbs(combined_back, alpha=(255.0))
    
    # Merge the channels back and convert to BGR
    processed_image = cv2.merge((combined_back, channels[1], channels[2]))
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_YCrCb2BGR)
    processed_image = (processed_image).astype(image.dtype)

    return processed_image


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
		source_img = preprocess_hazy_image(source_img)
		
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
