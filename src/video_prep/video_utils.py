from local_utils import inside
from tqdm import tqdm
import os, json

def getNumOverlaps(faceTrack, bodyTrack, fps):
    faceTrack = {int(round(b[0]*fps)): b for b in faceTrack}
    bodyTrack = {int(round(b[0]*fps)): b for b in bodyTrack}
    numOverlaps = 0
    for ts, faceBox in faceTrack.items():
        if ts in bodyTrack.keys():
            numOverlaps += inside(faceBox[1:5], bodyTrack[ts][1:5]) 
    return numOverlaps

def getShotWiseTracks(tracks):
    shotWiseTracks = {}
    for track in tracks.keys():
        shotId = track.split('_')[0]
        if shotId in shotWiseTracks.keys():
            shotWiseTracks[shotId].append(track)
        else:
            shotWiseTracks[shotId] = [track]
    return shotWiseTracks

def mapBodyFace(faceTracks, bodyTracks, framesObj):
    faceBodyMap = {} # for every face there will be body track
    shotWiseFaceTracks = getShotWiseTracks(faceTracks)
    shotWiseBodyTracks = getShotWiseTracks(bodyTracks)

    # TODO: Combine body tracks when they have same face track associated
    # In each shot, assigning a bodyTrack to each FaceTrack
    for shotId, faceTrackIds in tqdm(shotWiseFaceTracks.items()):
        if shotId not in shotWiseBodyTracks.keys():
            continue
        bodyTrackIds = shotWiseBodyTracks[shotId]
        for faceTrack in faceTrackIds:
            overlaps = []
            for bodyTrack in bodyTrackIds:
                overlaps.append([bodyTrack, getNumOverlaps(faceTracks[faceTrack], \
                                                            bodyTracks[bodyTrack], framesObj['fps'])])
            if max(overlaps, key=lambda x:x[1])[1] > 2:
                faceBodyMap[faceTrack] = max(overlaps, key=lambda x:x[1])[0]
    
    return faceBodyMap
    
def combineFaceTracks(faceTracks, faceBodyMap, faceTracksGroup):
    print(f'combined facetracks {faceTracksGroup} into one')
    # combine the faces of the tracks
    bodyTrack = faceBodyMap[faceTracksGroup[0]]
    combined = []
    for faceTrack in faceTracksGroup:
        combined.extend(faceTracks[faceTrack])
        faceTracks.pop(faceTrack)
        faceBodyMap.pop(faceTrack)
    faceTracks[faceTracksGroup[0]] = combined
    faceBodyMap[faceTracksGroup[0]] = bodyTrack
    return faceTracks, faceBodyMap

def remove_redundant_tracks(faceTracks, bodyTracks, faceBodyMap):
    # combine the face tracks which are associated with same body tracks. 
    shotWiseFaceTracks = getShotWiseTracks(faceTracks)
    shotWiseBodyTracks = getShotWiseTracks(bodyTracks)
    for shotId, faceTrackIds in shotWiseFaceTracks.items():
        if shotId in shotWiseBodyTracks.keys():
            bodyFaceTracksMaps = {}
            for faceTrack in faceTrackIds:
                if faceTrack in faceBodyMap.keys():
                    bodyTrack = faceBodyMap[faceTrack]
                    if bodyTrack in bodyFaceTracksMaps.keys():
                        bodyFaceTracksMaps[bodyTrack].append(faceTrack)
                    else:
                        bodyFaceTracksMaps[bodyTrack] = [faceTrack]
            for faceTracksGroup in bodyFaceTracksMaps.values():
                if len(faceTracksGroup) > 1:
                    faceTracks, faceBodyMap = combineFaceTracks(faceTracks, faceBodyMap, faceTracksGroup)
    return faceTracks, bodyTracks, faceBodyMap

def combine_track_names(faceTracks, bodyTracks, faceBodyMap):
    # remove the unassociated body tracks
    for trackId in bodyTracks.keys():
        if trackId not in faceBodyMap.values():
            bodyTracks[trackId].pop()
    
    faceTracksOut = {}
    for faceTrackId, faceTrack in faceTracks.items():
        if faceTrackId in faceBodyMap.keys():
            faceTracksOut[faceBodyMap[faceTrackId]] = faceTrack
        else:
            faceTracksOut[faceTrackId] = faceTrack
    return faceTracksOut, bodyTracks

def constrain_body_tracks(faceTracks, bodyTracks, fps):
    # contrain the body bbox to below the face]
    bodyTracksOut = {}
    for faceTrackId, faceTrack in faceTracks.items():
        frame_wise_faces = {int(round(box[0]*fps)): box for box in faceTrack}
        if faceTrackId in bodyTracks.keys():
            bodyTrack = bodyTracks[faceTrackId]
            bodyTrackOut = []
            for box in bodyTrack:
                ts, body_x1, body_y1, body_x2, body_y2 = box[:5]
                frame = int(round(fps*ts))
                if frame in frame_wise_faces.keys():
                    face_x1, face_y1, face_x2, face_y2 = frame_wise_faces[frame][1:5]
                    # restricting the body height to max of 4x the face height
                    # body_y2 = min(body_y2, face_y2 + 3*(face_y2 - face_y1))
                    # body_y1 = max(body_y1, face_y2) # body starts below the face
                    # face_center_x = (face_x1 + face_x2)/ 2
                    # body_x1 = max(body_x1 ,face_center_x - 1.5*(face_x2 - face_x1))
                    # body_x2 = min(body_x2, face_center_x + 1.5*(face_x2 - face_x1))
                    bodyTrackOut.append([ts, body_x1, body_y1, body_x2, body_y2] + box[5:])
                else:
                    bodyTrackOut.append(box)
            bodyTracksOut[faceTrackId] = bodyTrackOut
    for bodyTrackId, bodyTrack in bodyTracks.items():
        if bodyTrackId not in faceTracks.keys():
            bodyTracksOut[bodyTrackId] = bodyTrack
    return bodyTracksOut
    
def body_face_consistency(faceTracks, bodyTracks, framesObj):
    faceBodyMap = mapBodyFace(faceTracks, bodyTracks, framesObj)
    faceTracks, bodyTracks, faceBodyMap = remove_redundant_tracks(faceTracks, bodyTracks, faceBodyMap)
    bodyFaceMap = mapBodyFace(bodyTracks, faceTracks, framesObj)
    bodyTracks, faceTracks, bodyFaceMap = remove_redundant_tracks(bodyTracks, faceTracks, bodyFaceMap)
    faceBodyMap = mapBodyFace(faceTracks, bodyTracks, framesObj)
    faceTracks, bodyTracks, faceBodyMap = remove_redundant_tracks(faceTracks, bodyTracks, faceBodyMap)
    faceTracks, bodyTracks = combine_track_names(faceTracks, bodyTracks, faceBodyMap)
    bodyTracks = constrain_body_tracks(faceTracks, bodyTracks, framesObj['fps'])
    return faceTracks, bodyTracks