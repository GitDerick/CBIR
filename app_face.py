import cv2
import numpy as np
from distances import retrieve_similar_images
from data_processing import extract_features

signatures_glcm = np.load('./Signatures/FaceSignatures.npy')

def capture_images(num_images):
    captured_images = []
    cap = cv2.VideoCapture(0)
    for _ in range(num_images):
        ret, frame = cap.read()
        if ret:
            captured_images.append(frame)
    cap.release()
    return captured_images

def calculate_image_quality(image):
    # Calculate image quality (you may replace this with your own quality measure)
    # For demonstration, we'll use image size as a simple proxy for quality
    return image.shape[0] * image.shape[1]

def select_best_quality_image(images):
    best_quality = 0
    best_image = None
    for image in images:
        quality = calculate_image_quality(image)
        if quality > best_quality:
            best_quality = quality
            best_image = image
    return best_image

def main():
    # Capture multiple images from camera
    captured_images = capture_images(num_images=500)

    if captured_images:
        # Select the image with the best quality
        best_image = select_best_quality_image(captured_images)

        # Save the best quality image for later reference
        cv2.imwrite('best_quality_image.png', best_image)

        # Extract features from the best quality image
        features = extract_features('best_quality_image.png')

        # Retrieve similar images
        result = retrieve_similar_images(features_db=signatures_glcm, query_features=features, distance='canberra', num_results=10)
        print(f'Results\n-----------------------------\n{result}')
    else:
        print("Failed to capture images from camera")

if __name__=='__main__':
    main()