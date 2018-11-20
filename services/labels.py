import requests
import logging
from .interfaces import ILabels

class LabelsService(ILabels):
    MOODS = [
        'courageousAndAdventurous',
        'cozyWarmAndSafe',
        'darkAndStormy',
        'energeticAndIntense',
        'exploringAndCurious',
        'happyAndJoyful',
        'proudAndAmbitious',
        'relaxedAndPeaceful',
        'romanticAndSensitive',
        'sadAndReflective',
    ]

    def adjust_output(self, output):
        
        ret = {x[0]: x[1] for x in list(zip(self.MOODS, output))}

        return ret