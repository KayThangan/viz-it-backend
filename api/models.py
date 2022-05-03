from django.db import models
import sys

# Root directory of the project
ROOT_DIR = "../"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

class ImageSearch:
    image = ""
    result_image = sys.path[0] + "/results/result.png"
    image_datas = []

    def set_image(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def get_result_image(self):
        return self.result_image

    def get_image_datas(self):
        return self.image_datas

    def add_image_data(self, data):
        self.image_datas.append(data)

    def empty_image_datas(self):
        self.image_datas.clear()


class ImageData:
    object_class = ""
    coordinates = []

    def set_object_class(self, object_class):
        self.object_class = object_class

    def get_object_class(self):
        return self.object_class

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates

    def get_coordinates(self):
        return self.coordinates


image_search = ImageSearch()
