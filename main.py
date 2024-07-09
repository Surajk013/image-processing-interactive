import subprocess
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFilter
import os

def check_imagemagick():
    try:
        subprocess.run(["magick", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        print("ImageMagick is not installed or not found in your PATH.")
        return False

def resize_image_opencv(input_file, output_file, scale_factor=2):
    image = cv2.imread(input_file)
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_file, resized_image)
    print(f"Image size increased and saved to {output_file} using OpenCV")

def resize_image_pillow(input_file, output_file, scale_factor=2):
    image = Image.open(input_file)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size, Image.ANTIALIAS)
    image.save(output_file)
    print(f"Image size increased and saved to {output_file} using Pillow")

def correct_pixelation_opencv(input_file, output_file):
    image = cv2.imread(input_file)
    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_file, image)
    print(f"Pixelated image corrected and saved to {output_file} using OpenCV")

def correct_pixelation_pillow(input_file, output_file):
    image = Image.open(input_file)
    small = image.resize((image.width // 2, image.height // 2), resample=Image.BILINEAR)
    result = small.resize(image.size, Image.BILINEAR)
    result.save(output_file)
    print(f"Pixelated image corrected and saved to {output_file} using Pillow")

def detect_and_correct_pixels_opencv(input_file, output_file):
    image = cv2.imread(input_file)
    image = cv2.medianBlur(image, 3)
    cv2.imwrite(output_file, image)
    print(f"Inaccurate pixels detected and corrected, saved to {output_file} using OpenCV")

def detect_and_correct_pixels_pillow(input_file, output_file):
    image = Image.open(input_file)
    result = image.filter(ImageFilter.MedianFilter(size=3))
    result.save(output_file)
    print(f"Inaccurate pixels detected and corrected, saved to {output_file} using Pillow")

def sharpen_image_opencv(input_file, output_file):
    image = cv2.imread(input_file)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(output_file, sharpened)
    print(f"Image sharpened and saved to {output_file} using OpenCV")

def sharpen_image_pillow(input_file, output_file):
    image = Image.open(input_file)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    image.save(output_file)
    print(f"Image sharpened and saved to {output_file} using Pillow")

def enhance_colors_opencv(input_file, output_file):
    image = cv2.imread(input_file)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.5
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_file, enhanced_image)
    print(f"Colors enhanced and saved to {output_file} using OpenCV")

def enhance_colors_pillow(input_file, output_file):
    image = Image.open(input_file)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.5)
    image.save(output_file)
    print(f"Colors enhanced and saved to {output_file} using Pillow")

def show_comparison(input_file, output_opencv, output_pillow):
    from matplotlib import pyplot as plt

    input_image = Image.open(input_file)
    opencv_image = Image.open(output_opencv)
    pillow_image = Image.open(output_pillow)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(input_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(opencv_image)
    axs[1].set_title('Processed with OpenCV')
    axs[1].axis('off')

    axs[2].imshow(pillow_image)
    axs[2].set_title('Processed with Pillow')
    axs[2].axis('off')

    plt.show()

def main():
    print("Welcome to the Image Processing Interactive Session!\n")
    print("This script can help you process pixelated images and improve their quality.")
    print("Note: This script can work on any scale, but your system needs to have enough CPU power.\n")

    input_file = "example_input_image.jpg"  # Replace with your actual input image path
    output_opencv = "output_opencv.jpg"
    output_pillow = "output_pillow.jpg"

    options = [
        "1. Increase the size of the image",
        "2. Correct pixelated images",
        "3. Detect and correct inaccurate pixels",
        "4. Sharpen the image",
        "5. Enhance colors",
        "6. Exit"
    ]

    def show_options():
        print("\nChoose an option to process your image:")
        for option in options:
            print(option)

    imagemagick_available = check_imagemagick()

    while True:
        show_options()
        choice = input("Enter the number of your choice: ")

        if choice == '1':
            if imagemagick_available:
                resize_image_opencv(input_file, output_opencv)
            resize_image_pillow(input_file, output_pillow)

        elif choice == '2':
            if imagemagick_available:
                correct_pixelation_opencv(input_file, output_opencv)
            correct_pixelation_pillow(input_file, output_pillow)

        elif choice == '3':
            if imagemagick_available:
                detect_and_correct_pixels_opencv(input_file, output_opencv)
            detect_and_correct_pixels_pillow(input_file, output_pillow)

        elif choice == '4':
            if imagemagick_available:
                sharpen_image_opencv(input_file, output_opencv)
            sharpen_image_pillow(input_file, output_pillow)

        elif choice == '5':
            if imagemagick_available:
                enhance_colors_opencv(input_file, output_opencv)
            enhance_colors_pillow(input_file, output_pillow)

        elif choice == '6':
            print("Exiting the Image Processing Interactive Session. Goodbye!")
            break

        else:
            print("Invalid choice. Please select a valid option.")

        show_comparison(input_file, output_opencv, output_pillow)

if __name__ == "__main__":
    main()
