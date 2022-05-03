import base64
import sys

import skimage
from rest_framework.decorators import api_view
from rest_framework.response import Response

import tensorflow as tf
from tensorflow.python.keras.backend import get_session

import neural_network.models.mrcnn.model as modellib

from image_prediction import Prediction
from neural_network.models.mrcnn import visualise, utils
from .models import image_search, ImageData

prediction = Prediction()

global session
session = get_session()

# to load the model and save it for the entire environment use graph
global graph
graph = tf.compat.v1.get_default_graph()
# Create model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          model_dir=prediction.MODEL_DIR,
                          config=prediction.config)
# Set weights file path
weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

testing_neural_network = False

# Root directory of the project
ROOT_DIR = "../"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

@api_view(['GET'])
def getRoutes(request):
    routes = [
        {
            'Endpoint': '/image-search/upload/',
            'method': 'POST',
            'body': {'image': ""},
            'description': 'Upload a filename to local'
        },
        {
            'Endpoint': '/image-search/result/',
            'method': 'GET',
            'body': None,
            'description': 'Return a filename as result'
        },
        {
            'Endpoint': '/image-search/datas/',
            'method': 'GET',
            'body': None,
            'description': 'Return list of ImageData objects'
        },
    ]
    return Response(routes)

@api_view(['POST'])
def uploadImageSearch(request):
    image_search.empty_image_datas()

    data = request.data
    with open("./imageToTest.png", "wb") as fh:
        fh.write(base64.b64decode(data['image']))
    image_search.set_image("./imageToTest.png")

    print("xxxxxxxxxxxxxx", image_search.get_image())
    global session
    global graph
    with session.as_default():
        with graph.as_default():
            run_prediction(model)
    return Response("Image is uploaded!")

@api_view(['GET'])
def getImageResult(request):
    with open(image_search.get_result_image(), "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    result = {'result_image': encoded_string}
    return Response(result)

@api_view(['GET'])
def getImageDatas(request):
    result = []
    for object in image_search.get_image_datas():
        result.append(
            {
                'object_class': object.get_object_class(),
                'y1': object.get_coordinates()[0],
                'x1': object.get_coordinates()[1],
                'y2': object.get_coordinates()[2],
                'x2': object.get_coordinates()[3],
            }
        )
    return Response({"datas": result})

def run_prediction(model):
    image = skimage.io.imread(image_search.get_image())
    # Run object detection
    results = model.detect([image], verbose=0)

    # Display results
    r = results[0]

    visualise.display_instances(image, r['rois'], r['masks'],
                                r['class_ids'],
                                prediction.dataset_classes, r['scores'],
                                ax=None,
                                title="Predictions")

    for x in range(len(r['class_ids'])):
        data = ImageData()
        data.set_object_class(
            prediction.dataset_classes[r['class_ids'][x]])
        data.set_coordinates(r['rois'][x])
        image_search.add_image_data(data)

    if(testing_neural_network):
        test_network(model, image)


def test_network(model, image):
    prediction.proposal_classification(model, image)
    prediction.generating_masks(model, image)
