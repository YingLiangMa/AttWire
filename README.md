# AttWire Multiple Object detection: Detect both devices and catheters in images

Object detection using center point detection

## Highlights

- **Simple:** One-sentence method summary: use keypoint detection technic to detect the bounding box center point and regress to all other object properties like bounding box size, bounding box rotation, object centers.

- **Versatile:** The same framework works for object detection and catheter detection.

- **Fast:** The whole process in a single network feedforward. No NMS post processing is needed. Our  model runs at *55* FPS.

- **Strong**: Our best single model achieves *80.1*AP.


