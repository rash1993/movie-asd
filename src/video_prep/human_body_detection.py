from ultralyticsplus import YOLO, render_result
model = YOLO('ultralyticsplus/yolov8s')
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000
import cv2
vid =cv2.VideoCapture('/data/rash/talking_face/test_cases/ANDI_MACK_S03E03/sample4.mp4')
flag = True
frames = []
counter = 0
while flag:
    flag, img = vid.read()
    if flag:
        # print(counter)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

print(len(frames))
results = model.predict(frames, verbose=False, device=3)
# print(results[0].boxes.cls.cpu())
# print(results[0].boxes.xyxy.cpu())
# counter += 1
for i, result in enumerate(results):
    for box in result:
        box = box.boxes.cpu() 
        xyxy = box.xyxy[0].numpy()
        cls = box.cls[0].numpy()
        conf = box.conf[0].numpy()
        if int(cls) != 0:
            continue
        cv2.rectangle(frames[i], (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color=(255, 0, 0), thickness=2)
        cv2.putText(frames[i], str(cls), (int(xyxy[0]), int(xyxy[1]) - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), thickness=1, fontScale=2)
        cv2.putText(frames[i], str(conf), (int(xyxy[0]), int(xyxy[3]) + 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), thickness=1, fontScale=2)

save_path = 'body_test.mp4'
video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), \
                                   25, (int(frames[0].shape[1]), int(frames[0].shape[0])))
for frame in frames:
    video_writer.write(frame)
video_writer.release()