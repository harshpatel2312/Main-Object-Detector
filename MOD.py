import numpy as np
from ultralytics import YOLO

def pick_main_object(img):
    # img_arr = np.array(img)
    # print(img_arr.shape)
    
    # Load the model
    model = YOLO("yolo11n.pt")
    result = model(img)
    print(f"Bounding boxes: {result[0].boxes}")

    xywh = result[0].boxes.xywh
    
    # Area of object
    H, W = result[0].boxes.orig_shape
    widths = xywh[:, 2].numpy()
    heights = xywh[:, 3].numpy()
    
    areas = widths * heights
    areas_norm = areas / (W * H)

    # Confiddence
    conf = result[0].boxes.conf.numpy()
    
    # centre distance from the box
    H, W = result[0].boxes.orig_shape # Original shape
    
    # Image centre
    cx_img, cy_img = W / 2, H / 2
    
    # box centres
    cx = xywh[:, 0].numpy()
    cy = xywh[:, 1].numpy()
    
    # Distance formula (Euclidean)
    dx = cx - cx_img
    dy = cy - cy_img
    dist = np.sqrt(dx**2 + dy**2)
    
    # Normalization (max_dist = diagonal/2)
    max_dist = np.sqrt((W/2)**2 + (H/2)**2)
    dist_norm = dist/max_dist

    print("Centre distance: ", dist)
    print("Normalized distance: ", dist_norm)
    
    """
        Filtering main object
    
        Factor importance:
            alpha ~ Area of object
            beta ~ Confindence
            gamma ~ centre distance
    """
    
    alpha = 0.5
    beta = 0.4
    gamma = 0.1
    
    score = alpha*areas_norm + beta*conf - gamma*dist_norm
    main_idx = int(np.argmax(score))
    
    return main_idx, result