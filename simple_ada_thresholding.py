import cv2
import numpy as np
import matplotlib.pyplot as plt




# MANUAL IMPLEMENTATION



import numpy as np
import cv2
import matplotlib.pyplot as plt


def simple_thresholding(image, threshold_value):
    
    binary_image = np.where(image >= threshold_value, 255, 0).astype(np.uint8)
    return binary_image


def adaptive_thresholding(image, block_size, constant_value):
   
    # Initialize an output image with zeros (black)
    binary_image = np.zeros_like(image, dtype=np.uint8)

    # Iterate over each pixel in the image
    half_block = block_size // 2
    for i in range(half_block, image.shape[0] - half_block):
        for j in range(half_block, image.shape[1] - half_block):
            # Extract local neighborhood
            block = image[i - half_block : i + half_block + 1, j - half_block : j + half_block + 1]
            
            # Compute the mean of the neighborhood
            local_mean = np.mean(block)
            
            # Apply adaptive thresholding
            threshold_value = local_mean - constant_value
            binary_image[i, j] = 255 if image[i, j] >= threshold_value else 0

    return binary_image


image_path = r"C:\Users\path\Desktop\jupyter_extension_prettier\vs_code_extension\filtering_threshold_otsu_watershed_images_region_growing\train_023.png"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)



# Simple Thresholding
simple_thresh_value = 85
simple_binary_image = simple_thresholding(image, simple_thresh_value)

# Adaptive Thresholding
block_size = 11  # odd block size
constant_value = 3  # Constant subtracted from the mean of the neighborhood
adaptive_binary_image = adaptive_thresholding(image, block_size, constant_value)



# Display results using matplotlib
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Simple Thresholded image
plt.subplot(1, 3, 2)
plt.title("Simple Thresholding")
plt.imshow(simple_binary_image, cmap='gray')
plt.axis('off')

# Adaptive Thresholded image
plt.subplot(1, 3, 3)
plt.title("Adaptive Thresholding")
plt.imshow(adaptive_binary_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
































# LIBRARY IMPLEMENTATION


# def apply_segmentation_techniques(image_path):

#     img = cv2.imread(image_path)

#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # 1. Simple Thresholding
#     ret, simple_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
#     # 2. Adaptive Thresholding
#     adaptive_thresh = cv2.adaptiveThreshold(
#         gray, 255, 
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY, 
#         11, 2
#     )
    
#     # Display results
#     plt.figure(figsize=(12,8))
    
#     # Original Image
#     plt.subplot(2,2,1)
#     plt.title('Original Image')
#     plt.imshow(img_rgb)
#     plt.axis('off')
    
#     # Simple Thresholding
#     plt.subplot(2,2,2)
#     plt.title('Simple Thresholding')
#     plt.imshow(simple_thresh, cmap='gray')
#     plt.axis('off')
    
#     # Adaptive Thresholding
#     plt.subplot(2,2,3)
#     plt.title('Adaptive Thresholding')
#     plt.imshow(adaptive_thresh, cmap='gray')
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()

#     return {
#         'simple_threshold': simple_thresh,
#         'adaptive_threshold': adaptive_thresh
#     }

# # Example usage
# if __name__ == "__main__":
#     # Replace with your image path
#     image_path = r'C:\Users\path\Desktop\CV_DL_Practicals\filtering_threshold_otsu_watershed_images_region_growing\train_004.png'
#     results = apply_segmentation_techniques(image_path)