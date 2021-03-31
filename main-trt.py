import time
import json
import argparse

import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import PIL.Image

import trt_pose.coco
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

OPTIMIZED_MODEL = 'pose_model/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
IMAGE_SHAPE = 224

model_trt = TRTModule()

parse_objects = ParseObjects(topology, cmap_threshold=0.25, link_threshold=0.25)
draw_objects = DrawObjects(topology)

# Tracker.
max_age = 30
tracker = Tracker(max_age=max_age, n_init=3)

# Actions Estimate.
action_model = TSSTG()

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.resize(image, (IMAGE_SHAPE, IMAGE_SHAPE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def execute(image, only_skeleton = False):
    start = time.time()
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    # print(cmap.shape)
    # print('paf', paf.shape)
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    end = time.time()
    fps = 1/(end-start)
    # print('objects shape', objects.shape)
    # print('peaks shape', peaks.shape)
    if only_skeleton:
        image = np.zeros(image.shape)
    draw_objects(image, counts, objects, peaks)
    cv2.putText(image, 'FPS: %f' % (fps),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # print(counts
    if counts > 0:
        kpts_face, kpts_body, confs = get_keypoints(image, counts, objects, peaks)
        bboxes_body = kpt2bbox(kpts_body, ex=10)
        bboxes_face = face_kpt2bbox(kpts_face)

        # print('num faces: ', len(bboxes_face))
        for bbox in bboxes_face:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        # print(bbox)
        # for bbox in bboxes_body:
        #     cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 100, 255), 2)


        tracker.predict()
        # Create Detections object.
        detections = [Detection(bboxes_body[i].astype(np.float32),
                                np.concatenate((kpts_body[i], confs[i]), axis=1),
                               .9) for i in range(len(bboxes_body))]
        tracker.update(detections)
        perform_action_recognition(image)
    return image

def perform_action_recognition(frame):
    # Predict Actions of each track.
    for i, track in enumerate(tracker.tracks):
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_tlbr().astype(int)
        center = track.get_center().astype(int)

        action = 'Detecting..'
        clr = (0, 0, 255)
        # Use 30 frames time-steps to prediction.
        if len(track.keypoints_list) == 30:
            pts = np.array(track.keypoints_list, dtype=np.float32)
            # print(pts.shape)
            # print(frame.shape[:2])
            out = action_model.predict(pts, frame.shape[:2])
            action_name = action_model.class_names[out[0].argmax()]
            action = '{}'.format(action_name)
            if action_name == 'Fall Down':
                clr = (255, 0, 0)
            elif action_name == 'Lying Down':
                clr = (255, 200, 0)
            elif action_name == 'Detecting..':
                clr = (0, 0, 255)
            else:
                clr = (0, 255, 0)

        # VISUALIZE.
        if track.time_since_update == 0:
            # if args.show_skeleton:
                # frame = draw_single(frame, track.keypoints_list[-1])
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                0.6, (255, 0, 255), 2)
            frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (0, 0, 0), 1)
        return frame


def get_keypoints(image, object_counts, objects, normalized_peaks):
    all_kpt_face = []
    all_kpt_body = []
    all_confs = []
    height = image.shape[0]
    width = image.shape[1]

    count = int(object_counts[0])
    # print('count',  count)
    K = topology.shape[0]
    for i in range(count):
        face_keypoints = []
        body_keypoints = []
        confs = []
        obj = objects[0][i]
        C = obj.shape[0]
        for j in range(C):
            k = int(obj[j])
            # print(k)
            peak = normalized_peaks[0][j][k]
            x = round(float(peak[1]) * width)
            y = round(float(peak[0]) * height)
            if (j>0 and j<5) or j==17:
                if k>0:
                    face_keypoints.append([x,y])
            else:
                if k >=0 :
                    confs.append([.99])
                else:
                    confs.append([0.01])
                body_keypoints.append([x,y])

        all_kpt_face.append(np.array(face_keypoints))
        all_kpt_body.append(np.array(body_keypoints))
        all_confs.append(np.array(confs))
    return all_kpt_face, all_kpt_body, all_confs


def kpt2bbox(all_kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    bboxes = []
    for kpt in all_kpt:
        non_zero_kpt = []
        for (x, y) in kpt:
            if not (x == y == 0):
                non_zero_kpt.append([x,y])
        kpt = np.array(non_zero_kpt)

        if kpt.shape[0] > 4:
            bboxes.append(np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                        kpt[:, 0].max() + ex, kpt[:, 1].max() + ex)))
    return bboxes

def face_kpt2bbox(all_kpt):
    bboxes = []
    for kpt in all_kpt:
        if kpt.shape[0] > 0:
            top_left_x = kpt[:, 0].min()
            top_left_y = kpt[:, 1].min()
            bottom_right_x = kpt[:, 0].max()
            bottom_right_y = kpt[:, 1].max()
            ex = (bottom_right_x - top_left_x)//3
            bboxes.append(np.array((top_left_x-ex//2, top_left_y-ex, bottom_right_x+ex//2, bottom_right_y-ex)))
    return bboxes


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='TRT pose Demo.')
    par.add_argument('-C', '--camera', default='0', required=False,
                     help='Source of camera or video file path.')
    par.add_argument('-M', '--model', default='resnet', required=False,
                     help='Model to use')
    par.add_argument('--save_out', type=str, default='', required=False,
                     help='Model to use')
    args = par.parse_args()
    if args.camera == '0':
        camSet = 'v4l2src device=/dev/video0 ! video/x-raw,width=800,height=600, framerate=24/1 ! videoconvert ! appsink'
        cap = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)
    else:
        print(args.camera)
        if args.camera.split('.')[1] == 'mp4':
            camSet = f'filesrc location={args.camera} ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx, width=1280,height=720 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink'
            cap = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(args.camera)
    if args.model == 'densenet':
        # global OPTIMIZED_MODEL, IMAGE_SHAPE
        OPTIMIZED_MODEL = 'pose_model/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        IMAGE_SHAPE = 256
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_out, codec, 25, (320, 240))
    else:
        outvid = False
    

    if not cap.isOpened():
            print("Cannot open camera\n")
            exit(1)

    while True:
            ret, img = cap.read()
            if not ret:
                    print("Stream end\n")
                    break

            # cv2.imshow('img', img)
            image_w = execute(img)
            cv2.imshow('out', image_w)
            if outvid:
                writer.write(image_w)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                    break

    cap.release()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
