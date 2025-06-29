import cv2

img = cv2.imread(r"C:\Users\Admin\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1\asl_alphabet_train\asl_alphabet_train\B\B1.jpg")
print("Image shape:", img.shape)  # (height, width, channels)
print("Height:", img.shape[0])
print("Width:", img.shape[1])