"""
Image Noise Reduction using Singular Value Decomposition (SVD)
Author: Muhammad Ahmed
Date: 2023-3-11

Description:
This script demonstrates how to remove noise from an image using SVD. The Method implemented here is
Hard Thresholding by Matan Gavish and David L. Donoho in their paper called,
"The Optimal Hard Threshold for Singular Values is 4/√3 " please read that.
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Check if the image file exists
image_path = 'iPhone-15-Pro-Burgandy-Feature-2.jpg'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image '{image_path}' not found.")

# Open and convert the image to grayscale
image = Image.open(image_path).convert('L')
# Convert the image to a NumPy array
image_array = np.array(image)
print(f'image rank is {np.linalg.matrix_rank(image_array)}')

# Display the original image
plt.set_cmap('gray')
plt.imshow(image_array)
plt.title('ORIGINAL IMAGE')
plt.show()

# Add noise to the original image
sigma = 30
imagenoisy = image_array + sigma*np.random.randn(*image_array.shape)

# Display the noisy image
plt.imshow(imagenoisy)
plt.axis('off')
plt.title("NOISY IMAGE")
plt.show()

# Remove noise using thresholding and SVD
U, S, VT = np.linalg.svd(imagenoisy, full_matrices=False)
N = imagenoisy.shape[0]
cutoff = (4/np.sqrt(3)) * np.sqrt(N) * sigma  # hard threshold
r = np.max(np.where(S > cutoff))
XClean = U[:, :(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1), :]

# Display the cleaned image
plt.imshow(XClean)
plt.axis('off')
plt.title("REMOVED NOISE AND RECOVERED THE IMAGE")
plt.show()

# Plot the singular values of the image
fig1, ax1 = plt.subplots(1)
ax1.semilogy(S, '-o', color='k', linewidth=2)
ax1.semilogy(np.diag(S[:(r+1)]), 'o', color='r', linewidth=2)
ax1.plot(np.array([-20, N+20]), np.array([cutoff, cutoff]), '--', color='r')
ax1.grid()
plt.show()

# Uncomment below to use an energy-based approach
# cds = np.cumsum(S) / np.sum(S)
# r90 = np.min(np.where(cds > 0.90))
# X90 = U[:, :(r90+1)] @ np.diag(S[:(r90+1)]) @ VT[:(r90+1), :]
# plt.imshow(X90)
# plt.title("Energy Based Approach")
# plt.show()
