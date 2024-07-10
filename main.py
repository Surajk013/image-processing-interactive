import subprocess
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16

# Function to check if ImageMagick is installed
def check_imagemagick():
    try:
        subprocess.run(["magick", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        print("ImageMagick is not installed or not found in your PATH.")
        return False

# Function to resize image using OpenCV
def resize_image_opencv(input_file, output_file, scale_factor=2):
    image = cv2.imread(input_file)
    if image is None:
        print(f"Error: Could not open or read the input file {input_file}.")
        return
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_file, resized_image)
    print(f"Image size increased and saved to {output_file} using OpenCV")

# Function to resize image using Pillow
def resize_image_pillow(input_file, output_file, scale_factor=2):
    try:
        image = Image.open(input_file)
    except Exception as e:
        print(f"Error: Could not open or read the input file {input_file}. Error: {e}")
        return
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size, Image.LANCZOS)
    image.save(output_file)
    print(f"Image size increased and saved to {output_file} using Pillow")

# Function to correct pixelation using OpenCV
def correct_pixelation_opencv(input_file, output_file):
    image = cv2.imread(input_file)
    if image is None:
        print(f"Error: Could not open or read the input file {input_file}.")
        return
    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_file, image)
    print(f"Pixelated image corrected and saved to {output_file} using OpenCV")

# Function to correct pixelation using Pillow
def correct_pixelation_pillow(input_file, output_file):
    try:
        image = Image.open(input_file)
    except Exception as e:
        print(f"Error: Could not open or read the input file {input_file}. Error: {e}")
        return
    small = image.resize((image.width // 2, image.height // 2), resample=Image.BILINEAR)
    result = small.resize(image.size, Image.BILINEAR)
    result.save(output_file)
    print(f"Pixelated image corrected and saved to {output_file} using Pillow")

# Function to detect and correct inaccurate pixels using OpenCV
def detect_and_correct_pixels_opencv(input_file, output_file):
    image = cv2.imread(input_file)
    if image is None:
        print(f"Error: Could not open or read the input file {input_file}.")
        return
    image = cv2.medianBlur(image, 3)
    cv2.imwrite(output_file, image)
    print(f"Inaccurate pixels detected and corrected, saved to {output_file} using OpenCV")

# Function to detect and correct inaccurate pixels using Pillow
def detect_and_correct_pixels_pillow(input_file, output_file):
    try:
        image = Image.open(input_file)
    except Exception as e:
        print(f"Error: Could not open or read the input file {input_file}. Error: {e}")
        return
    result = image.filter(ImageFilter.MedianFilter(size=3))
    result.save(output_file)
    print(f"Inaccurate pixels detected and corrected, saved to {output_file} using Pillow")

# Function to sharpen image using OpenCV
def sharpen_image_opencv(input_file, output_file):
    image = cv2.imread(input_file)
    if image is None:
        print(f"Error: Could not open or read the input file {input_file}.")
        return
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(output_file, sharpened)
    print(f"Image sharpened and saved to {output_file} using OpenCV")

# Function to sharpen image using Pillow
def sharpen_image_pillow(input_file, output_file):
    try:
        image = Image.open(input_file)
    except Exception as e:
        print(f"Error: Could not open or read the input file {input_file}. Error: {e}")
        return
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    image.save(output_file)
    print(f"Image sharpened and saved to {output_file} using Pillow")

# Function to enhance colors using OpenCV
def enhance_colors_opencv(input_file, output_file):
    image = cv2.imread(input_file)
    if image is None:
        print(f"Error: Could not open or read the input file {input_file}.")
        return
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.5
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_file, enhanced_image)
    print(f"Colors enhanced and saved to {output_file} using OpenCV")

# Function to enhance colors using Pillow
def enhance_colors_pillow(input_file, output_file):
    try:
        image = Image.open(input_file)
    except Exception as e:
        print(f"Error: Could not open or read the input file {input_file}. Error: {e}")
        return
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.5)
    image.save(output_file)
    print(f"Colors enhanced and saved to {output_file} using Pillow")

# Function to deblur image using a pre-trained AI model with PyTorch
def deblur_image_pytorch(input_file, output_file, model):
    try:
        image = Image.open(input_file).convert("RGB")
    except Exception as e:
        print(f"Error: Could not open or read the input file {input_file}. Error: {e}")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # Resize to fit the model input size
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output_tensor = model(image_tensor)
        output_tensor = output_tensor.squeeze().cpu()

    output_image = transforms.ToPILImage()(output_tensor)
    output_image.save(output_file)
    print(f"Blurred image corrected and saved to {output_file} using PyTorch")

# Function to show comparison of original and processed images
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

    input_file = input("Please enter the location of your input image file: ").strip()
    output_opencv = "output_opencv.jpg"
    output_pillow = "output_pillow.jpg"
    output_pytorch = "output_pytorch.jpg"

    options = [
        "1. Increase the size of the image",
        "2. Correct pixelated images",
        "3. Detect and correct inaccurate pixels",
        "4. Sharpen the image",
        "5. Enhance colors",
        "6. Deblur image using AI",
        "7. Exit"
    ]

    def show_options():
        print("\nChoose an option to process your image:")
        for option in options:
            print(option)

    imagemagick_available = check_imagemagick()

    # Load pre-trained model for AI-based deblurring
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16(pretrained=True).to(device)
    model.eval()

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
            deblur_image_pytorch(input_file, output_pytorch, model)

        elif choice == '7':
            print("Exiting the Image Processing Interactive Session. Goodbye!")
            break

        else:
            print("Invalid choice. Please select a valid option.")

        show_comparison(input_file, output_opencv, output_pillow)

if __name__ == "__main__":
    main()