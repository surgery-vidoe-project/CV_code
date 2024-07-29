import cv2  # Importing the opencv library
import numpy as np  # Import the numpy library, often used to perform mathematical operations
import os  # Importing the os library for manipulating the file system
import argparse  # Importing the argparse library for parsing command line arguments

# Removing noise from images
def remove_noise(image):
    # Bilateral filtering is first used to remove noise from the image while keeping as much edge information as possible
    # d=12 indicates the diameter around the filter, sigmaColor=190 and sigmaSpace=190 determines the range and intensity of filtering
    image = cv2.bilateralFilter(image, 17, 200, 200)
    # Gaussian filtering is then used to remove the Gaussian noise
    image = cv2.GaussianBlur(image, (5,5), 0)
    # The noise is then further removed using non-local mean denoising
    # h=5 indicates the filter strength, templateWindowSize=7 and searchWindowSize=21 determine the denoising window size
    image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    # Create a sharpening filter kernel
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # Applied Sharpening Filter
    image = cv2.filter2D(image, -1, sharpen_kernel)
    return image  # Returns the denoised image

#1. Locate: find the black areas (look for areas where black is concentrated)
def find_black_circles(image):
    # Convert to HSV colour space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Sets the colour range of the black area
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 150])
    # Creating a Mask
    mask = cv2.inRange(hsv, lower_black, upper_black)
    return mask
#2. Fix missing areas in images python main.py xray_images
def inpaint_missing_regions(image):
    # Introduction of black areas
    mask = find_black_circles(image)
    # Use the inpaint method for image repair. inpaintRadius=7 defines the radius of the surrounding pixels to be taken into account when repairing the image.
    image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
    return image  # Returns the repaired image

# Adjusting the contrast and brightness of an image
def adjust_contrast_brightness(image):
    beta = -40
    alpha = 1.3
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image
# Application of CLAHE algorithm to enhance local contrast of images
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Creating CLAHE objects with custom clip limits and tile sizes
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # Check if the image is a colour image (3 colour channels)
    if len(image.shape) == 3 and image.shape[2] == 3:  # colour image
        channels = cv2.split(image)  # Separate channels for colour images
        # Apply CLAHE to each channel
        eq_channels = []  
        for ch in channels:
            eq_channels.append(clahe.apply(ch))
        image = cv2.merge(eq_channels)  # Merging CLAHE-enhanced channels
    else:  # If the image is a greyscale map (only one colour channel)
        image = clahe.apply(image)  # Apply CLAHE algorithm directly to grey scale images
    return image  # Returns the image after contrast enhancement

# Adjusting the colour balance of an image
def adjust_color_balance(image, red_gain=1.0, green_gain=1.0, blue_gain=1.0):
    (b, g, r) = cv2.split(image)   # Separate the BGR channel of the image
    # Adjust the intensity of each channel separately, a gain factor greater than 1 will enhance that colour, less than 1 will diminish it
    r = cv2.multiply(r, red_gain)  # Adjust the intensity of the red channel
    g = cv2.multiply(g, green_gain)  # Adjusting the intensity of the green channel
    b = cv2.multiply(b, blue_gain)  # Adjust the intensity of the blue channel
    return cv2.merge([b, g, r])  # Merge the adjusted channels back into one image

def correct_perspective(image):
    # Manually determine the source points, i.e. the coordinates of the four corner points of the distorted region in the image
    src_points = np.float32([[7, 15], [233, 6], [33, 234], [251, 227]])
    # Target point coordinates, usually the four corners of the image
    dst_points = np.float32([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]])
    # Calculating the Perspective Transformation Matrix
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # Applying Perspective Transformations
    corrected_image = cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))
    return corrected_image

# Processing single image files
def process_image(path):
    image = cv2.imread(path)  # Reading image files
    if image is None:  # Returns None if the read fails
        return None
    image = remove_noise(image)  # Denoising the image
    image = inpaint_missing_regions(image)  # Repair missing areas of the image
    image = adjust_contrast_brightness(image)  # Handling contrast and brightness
    image = adjust_color_balance(image, 1.1, 1.0, 0.9)  # Adjusting the colour balance
    image = apply_clahe(image)  # Applying CLAHE
    image = correct_perspective(image)   # Finally, apply a perspective transformation to correct the image
    return image  # Returns the processed image

# Main function that processes the image files of the whole directory and saves the processed results
def main(input_dir, output_dir):
    if not os.path.exists(output_dir):  # If the output directory does not exist, create it
        os.makedirs(output_dir)
    # Get all files ending in .jpg in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    # Loop through each file
    for file in files:
        img_path = os.path.join(input_dir, file)  # Get the full path of each file
        processed_image = process_image(img_path)  # process image
        if processed_image is not None:  # If processing is successful, the image is saved
            cv2.imwrite(os.path.join(output_dir, file), processed_image)

if __name__ == "__main__":
    # Using argparse to parse command line arguments
    parser = argparse.ArgumentParser(description='Process X-ray images.')
    parser.add_argument('input_dir', type=str, help='Directory containing X-ray images')
    parser.add_argument('output_dir', type=str, nargs='?', default='Results', help='Directory to save processed images')
    args = parser.parse_args()  # parsing parameter
    main(args.input_dir, args.output_dir)  # Call the main function to process the image