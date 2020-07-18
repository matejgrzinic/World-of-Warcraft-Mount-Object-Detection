import tensorflow as tf
from PIL import ImageGrab
import numpy as np
import cv2
import time
import core.utils as utils


input_size = 416
output_size = (1600, 900)

# model = tf.keras.models.load_model('models/yolov4-tiny-416')
model = tf.keras.models.load_model('models/wow')

while True:
    start_time = time.time()
    img = ImageGrab.grab() #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img_np = np.array(img) #this is the array obtained from conversion
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    
    images_data = [image_data]    
    images_data = np.asarray(images_data).astype(np.float32)
    
    infer = model.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
        
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.45
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(frame, pred_bbox)    
    
    result = np.asarray(image)
    result = cv2.resize(result, output_size)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result", result)
    
    fps = 1.0 / (time.time() - start_time)
    print("FPS: %.2f" % fps) 
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
    
cv2.destroyAllWindows()