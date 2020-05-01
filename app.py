import os
import sys
import cv2
from datetime import datetime
from pathlib import Path
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_cors import CORS, cross_origin

# TensorFlow and tf.keras
import tensorflow as tf

# Some utilites
import numpy as np
from util import base64_to_pil
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Declare a flask app
app = Flask(__name__)
CORS(app)


PATH_TO_FROZEN_GRAPH = "frozen_inference_graph.pb"
PATH_TO_LABEL_MAP = "label_map.pbtxt"
IMAGE_PATH = r"E:/GPP/keras-flask-deploy-webapp-master/static/images"
NUM_CLASSES = 7

#reads the frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

    
def model_predict(img):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # ret, image_np = cap.read()
            # image_np = cv2.imread(path) 
            # image_np = image
            image_np = img
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detections
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3,
                )
            # Display output
            # cv2.imshow('Waste Detection', cv2.resize(image_np, (800, 600)))
            prefix = datetime.now().strftime("%d%m%Y_%H%M%S_")
            filename = str(prefix)+"prediction.jpg"
            filepath = IMAGE_PATH+"/"+filename
            cv2.imwrite(filepath,image_np)
    return f"/static/images/{filename}"


@app.route('/image', methods=['GET'])
def image():
    # Main page
    return render_template('predict.html')


@app.route('/')
def home():
    return render_template('login2.html')


@app.route('/aboutus')
def about():
    return render_template('about.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("E:/GPP/keras-flask-deploy-webapp-master/uploads/image.jpg")

        # Make prediction
        img = cv2.imread("E:/GPP/keras-flask-deploy-webapp-master/uploads/image.jpg")
        # print(img)
        img_path = model_predict(img)
        
        # Serialize the result, you can add additional fields
        return jsonify({'image_url': img_path})

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('127.0.0.1', 5000), app)
    http_server.serve_forever()
