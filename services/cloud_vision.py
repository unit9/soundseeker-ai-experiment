from urllib.request import Request, urlopen
import json
from base64 import b64encode
from PIL import Image
from tools import pil_to_byteIO
from .interfaces import ICloudVision

class CloudVisionService(ICloudVision):
    """A class for interacting with Google Cloud Vision API."""

    API_PATH = 'https://vision.googleapis.com/v1/images:annotate?key='

    def __create_request_data(self, encoded_image):
        """
        " @param encoded_image b64encoded image
        """
        request_data = {
            'requests': [
              {
                'image': {
                  'content': encoded_image.decode('utf-8')
                },
                'features': [
                  {
                    'type': 'FACE_DETECTION',
                    'maxResults': 50
                  },
                  {
                    'type': 'LABEL_DETECTION',
                    'maxResults': 30
                  },
                  {
                    'type': 'LANDMARK_DETECTION',
                    'maxResults': 10
                  },
                  {
                    'type': 'SAFE_SEARCH_DETECTION',
                    'maxResults': 10
                  },
                  {
                    'type': 'IMAGE_PROPERTIES',
                    'maxResults': 10
                  },
                  {
                    'type': 'WEB_DETECTION',
                    'maxResults': 10
                  }
                ]
              }
            ]
        }
        return request_data

    def __init__(self, api_key):
        self.api_key = api_key                

    def recognize_features(self, img: Image):        
        """Run feature recognition on the provided image."""

        image = pil_to_byteIO(img)
        encoded_image = b64encode(image)
        request_data = self.__create_request_data(encoded_image)

        url = self.API_PATH + self.api_key
        body = json.dumps(request_data).encode('utf-8')
        request = Request(url, body, {'Content-Type': 'application/json'})
        with urlopen(request) as resp:
            data = resp.read() 
            
        response = json.loads(data.decode('utf-8'))
        return response['responses'][0]