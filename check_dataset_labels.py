
import cv2
import os

dataset_dir = "air_mnist_dataset/training"

for digit in range(10):
    digit_dir = os.path.join(dataset_dir, str(digit))
    for file in os.listdir(digit_dir):
        if file.endswith(".png"):
            path = os.path.join(digit_dir, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            cv2.imshow(f"Folder {digit}", img)
            key = cv2.waitKey(500)  # Show each image for 0.5 seconds
            if key == 27:  # ESC to quit early
                break
cv2.destroyAllWindows()
