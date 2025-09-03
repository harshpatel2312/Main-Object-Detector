import cv2
from ultralytics import YOLO
from MOD import pick_main_object

img = cv2.imread("images/test_14.jpeg")
cv2.imshow("Original Picture", img)

main_idx, result = pick_main_object(img)
print(f"Main Object Index: {main_idx}")

x1, y1, x2, y2 = result[0].boxes.xyxy[main_idx].numpy().astype(int)

img_vls = result[0].orig_img.copy()
cv2.rectangle(img_vls, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.imshow("Main Object", img_vls)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Predicted class names
model = YOLO("yolo11n.pt")
print(f"Detected Object: {model.names[int(result[0].boxes.cls[main_idx])]}")

# Drop image
cropped_img = img[y1:y2, x1:x2] # Use this to extarct the Region of Interest (ROI)
cv2.imshow("Detected Object", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()