import os
from PIL import Image

# Set the working directory to the ML folder to ensure that all file paths are relative to this directory.
os.chdir("C:/Users/prohi/PycharmProjects/POC/ML")

# Input and output paths
input_path = r"data/input/booking.jpeg"
output_path = r"data/output/booking.pdf"

# Open the image
image = Image.open(input_path)

# Convert to RGB (important for JPEGs)
image = image.convert("RGB")

# Save as PDF
image.save(output_path)
