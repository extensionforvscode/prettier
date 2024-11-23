import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_watershed_segmentation(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal using morphological operations
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(img, markers)
    
    # Mark boundaries in red
    img[markers == -1] = [255, 0, 0]
    
    return img, markers, thresh

def display_results(original, segmented, markers, binary):
    plt.figure(figsize=(15, 4))
    
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(binary, cmap='gray')
    plt.title('Otsu Binarization')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title('Watershed Segmentation')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(markers, cmap='jet')
    plt.title('Markers')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Replace with your image path
    image_path = r"C:\Users\path\Desktop\jupyter_extension_prettier\vs_code_extension\filtering_threshold_otsu_watershed_images_region_growing\train_038.png"
    
    # Read original image
    original = cv2.imread(image_path)
    
    # Apply watershed segmentation
    segmented, markers, binary = apply_watershed_segmentation(image_path)
    
    # Display results
    display_results(original, segmented, markers, binary)