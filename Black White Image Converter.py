import cv2
import math
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# === CONFIG ===
output_xmax = 200
output_ymax = 200
min_dist = 2.5  # Minimum spacing between points

# === OPEN FILE DIALOG ===
Tk().withdraw()  # Hide the root tkinter window
file_path = askopenfilename(title="Select a black-and-white image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])

if not file_path:
    print("No file selected. Exiting.")
    exit()

# === LOAD IMAGE ===
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
inverted = 255 - binary

# === FIND CONTOURS (INCLUDING NESTED) ===
contours, hierarchy = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

height, width = binary.shape
all_coordinates = []

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# === FILTER AND SCALE CONTOURS ===
for contour in contours:
    simplified_path = []
    prev_point = None

    for point in contour:
        x, y = point[0]

        # Normalize and scale
        x_scaled = (x / width) * output_xmax
        y_scaled = (y / height) * output_ymax

        # Flip origin to bottom-right
        x_final = output_xmax - x_scaled
        y_final = output_ymax - y_scaled

        current_point = (x_final, y_final)

        if prev_point is None or distance(current_point, prev_point) >= min_dist:
            simplified_path.append(current_point)
            prev_point = current_point

    # Close the contour loop
    if len(simplified_path) > 1:
        simplified_path.append(simplified_path[0])

    if simplified_path:
        all_coordinates.append(simplified_path)

# === PLOT THE CLOSED SIMPLIFIED CONTOURS ===
plt.figure(figsize=(8, 8))
plt.title("Closed Simplified Scaled Contours (Nested)")
plt.axis("equal")

# Plot each path
for path in all_coordinates:
    x_vals, y_vals = zip(*path)
    plt.plot(x_vals, y_vals, marker='o', markersize=3, linestyle='-')

# Flip Y-axis so (0,0) is at the bottom-right corner
plt.gca().invert_yaxis()

# Set X and Y labels
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# Set axis limits to match the plotter's range
plt.xlim(0, output_xmax)
plt.ylim(0, output_ymax)

# Show the plot with custom origin
plt.show()

print(all_coordinates)