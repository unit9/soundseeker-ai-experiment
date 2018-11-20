from abc import ABCMeta, abstractmethod
from PIL import Image

class ILabels:
    __metaclass__ = ABCMeta

    @classmethod
    def version(self): return "1.0"
    @abstractmethod
    def adjust_output(self, output): raise NotImplementedError



class ICloudVision:
    __metaclass__ = ABCMeta

    @classmethod
    def version(self): return "1.0"
    @abstractmethod
    def recognize_features(self, img: Image): raise NotImplementedError 