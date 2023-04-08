import cv2
import time
import numpy as np

with open('./cv_models/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# with open('./cv_models/classification_classes_ILSVRC2012.txt', 'r') as f:
#     image_net_names = f.read().split('\n')
# # final class names (just the first word of the many ImageNet names for one image)
# class_names = [name.split(',')[0] for name in image_net_names]

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model = cv2.dnn.readNet(model='./cv_models/frozen_inference_graph.pb',
                        config='./cv_models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')
# model = cv2.dnn.readNet(model='./cv_models/DenseNet_121.caffemodel',
#                         config='./cv_models/DenseNet_121.prototxt', 
#                         framework='Caffe')

# cap = cv2.VideoCapture('../../input/video_1.mp4')
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(
    './Captured/video_result.mp4', 
    cv2.VideoWriter_fourcc(*'mp4v'), 30, 
    (frame_width, frame_height))

def main():  # sourcery skip: remove-unnecessary-else, swap-if-else-branches
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image = frame
            image_height, image_width, _ = image.shape
            blob = cv2.dnn.blobFromImage(
                image=image, size=(300, 300), mean=(104, 117, 123), 
                swapRB=True)
            start = time.time()
            model.setInput(blob)
            output = model.forward()        
            end = time.time()
            fps = 1 / (end-start)
            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > .4:
                    # get the class id
                    class_id = detection[1]
                    # map the class id to the class 
                    class_name = class_names[int(class_id)-1]
                    color = COLORS[int(class_id)]
                    # get the bounding box coordinates
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    # get the bounding box width and height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    # draw a rectangle around each detected object
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                    # put the class name text on the detected object
                    cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    # put the FPS text on top of the frame
                    cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

            cv2.imshow('image', image)
            out.write(image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()