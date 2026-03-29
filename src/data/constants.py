
CLASSES = [
    "Aeroplane", 
    "Bicycle", 
    "Bird", 
    "Boat", 
    "Bottle",
    "Bus", 
    "Car", 
    "Cat", 
    "Chair", 
    "Cow",
    "Diningtable", 
    "Dog", 
    "Horse", 
    "Motorbike", 
    "Person",
    "Pottedplant", 
    "Sheep", 
    "Sofa", 
    "Train", 
    "Tvmonitor",
]

CLASS_TO_IDX = {class_name : class_id
                for class_id, class_name in enumerate(CLASSES)
}

CLASS_IDX_TO_NAME = {class_id : class_name
                     for class_name, class_id in CLASS_TO_IDX.items()
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]