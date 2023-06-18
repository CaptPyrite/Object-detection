from PIL import Image
from transformers import pipeline
import cv2
import numpy as np
from random import randrange

object_detector = pipeline("object-detection")
image = Image.open(r"Z:/Coding/image.jpg")
image_np = np.array(image)
results = object_detector(image)

for result in results:
    box = result["box"]
    label = result["label"]
    confidence = result["score"]
    
    x = int(box["xmin"])
    y = int(box["ymin"])
    
    w = int(box["xmax"]) - x
    h = int(box["ymax"]) - y
    
    rand_color = (randrange(255), randrange(255), randrange(255))
    
    cv2.rectangle(image_np, (x,y), (x + w, y + h), rand_color, 2)
    cv2.putText(image_np, f"Label: {label}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(image_np, f"Confidence: {confidence:.5f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    
cv2.imshow("Objects Found", image_np) 
cv2.waitKey(0)
cv2.destroyAllWindows()