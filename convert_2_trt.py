import json
import trt_pose.coco
import trt_pose.models

import torch
import torch2trt

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

# model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links, pretrained=False).cuda().eval()
model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links, pretrained=False).cuda().eval()

# MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'

model.load_state_dict(torch.load(MODEL_WEIGHTS))
print('loaded model')

WIDTH = 256
HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

# OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'

torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)