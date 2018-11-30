import time

import cv2 as cv
import tensorflow as tf

DINO_CLASS_ID = 1
OBSTACLE_CLASS_ID = 2
BIRD_CLASS_ID = 3
OVER_CLASS_ID = 4

graph_def = None


def setup_detection_environment():
    # Read the graph.
    with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.summary.FileWriter('logs', graph_def)

def run_detection(encoded_image):
    count = 1
    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        last_time = time.time()

        tensor_num_detections = sess.graph.get_tensor_by_name('num_detections:0')
        tensor_detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
        tensor_detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        tensor_detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')

        count = count + 1
        # capture frames
        img = cv.imdecode(encoded_image)
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        # cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))

        # Read and pre-process an image.
        # img = printscreen.copy()
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model

        out = sess.run([tensor_num_detections,
                        tensor_detection_scores,
                        tensor_detection_boxes,
                        tensor_detection_classes],
                       feed_dict={
                           'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        locs = {}
        obstacles = []
        for i in range(num_detections):
            idx = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]

            if score > 0.6:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                position = (int(x), int(y))
                cv.rectangle(img, position, (int(right), int(bottom)), (255, 0, 0), thickness=2)

                print(score, bbox[1], bbox[0], bbox[3], bbox[2])
                if idx == DINO_CLASS_ID:
                    locs["dino"] = (right, bottom)
                elif idx == OBSTACLE_CLASS_ID:
                    obstacles.append(position)
                elif idx == BIRD_CLASS_ID:
                    locs["bird"] = position

        minDist = float("Inf")
        for obstacle in obstacles:
            xDist = obstacle[0] - locs["dino"][0]
            print(xDist)
            if 0 <= xDist < minDist:
                minDist = xDist

        cv.imwrite('game-res' + str(count) + '.png', img)

