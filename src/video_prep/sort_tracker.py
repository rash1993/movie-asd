import numpy as np
from sort.sort import Sort

class Tracker:
    def __init__(self, tracker_type):
        self.tracker_type = tracker_type
        self.tracker = self.get_tracker(tracker_type)
        self.bbox = None
    
    def track(dets):
        