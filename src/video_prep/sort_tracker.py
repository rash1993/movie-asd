import numpy as np
from sort.sort import Sort

class SortTracker:
    def __init__(self, prefix='default'):
        self.tracker = Sort()
        self.track_name_prefix = prefix
    
    def track(self, dets):
        tracks = {}
        for frame_det in dets:
            time_stamp = frame_det[0][0]
            frame_det = np.array([det[1:] for det in frame_det])
            track_ids = self.tracker.update(frame_det)
            for track_id in track_ids:            
                bbox = [time_stamp] + track_id[:4].tolist()
                track_id = f'{self.track_name_prefix}_{int(track_id[4])}'
                if track_id in tracks:
                    tracks[track_id].append(bbox)
                else:
                    tracks[track_id] = [bbox]
        print(tracks.keys())
        return tracks