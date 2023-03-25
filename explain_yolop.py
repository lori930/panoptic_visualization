import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from PIL import Image
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])

# https://github.com/jacobgil/pytorch-grad-cam/issues/92#issuecomment-850448934
class YOLOPPrediction(torch.nn.Module):
    def __init__(self, model, task):
        super(YOLOPPrediction, self).__init__()
        self.model = model
        self.task = task
    
    def forward(self, x):
        # det_out, da_seg_out,ll_seg_out
        if self.task == 'det':
            return self.model(x)[0][0]
        elif self.task == 'da_seg':
            return self.model(x)[1]
        elif self.task == 'll_seg':
            return self.model(x)[2]
        else:
            return self.model(x)  
    
def detect(cfg,opt):
    
    dataset = LoadImages(opt.source, img_size=opt.img_size)

    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=opt.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(opt.device)

    cam_model = YOLOPPrediction(model, opt.task)
    
    target_layers = [cam_model.model.model[opt.layer]]
    cam = EigenCAM(cam_model, target_layers)
    
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        tensor = transform(img).to(opt.device)
        # print(tensor.shape)
        rgb_img = np.float32(tensor.permute(1,2,0).numpy()) / 255
        # print(rgb_img.shape)
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        # print(tensor.shape)
        det_out, da_seg_out,ll_seg_out = model(tensor)
        inf_out, _ = det_out
        
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        det=det_pred[0]
        _, _, height, width = tensor.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]
        
        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        
        if opt.label:
            img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

            if len(det):
                det[:,:4] = scale_coords(tensor.shape[2:],det[:,:4],img_det.shape).round()
                for *xyxy,conf,cls in reversed(det):
                    label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        
        # output = cam_model(tensor)
                
        # targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
        targets = [SemanticSegmentationTarget(category=18, mask=ll_seg_mask)]
        grayscale_cam = cam(tensor, targets=targets)[0, :, :]
        # cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        # grayscale_cam = cam(tensor, targets=targets)
        # grayscale_cam = grayscale_cam[0, :]
        # print(type(img_det), img_det.shape)
        cam_image = show_cam_on_image(np.float32(img_det) / 255, cv2.resize(grayscale_cam, (1280, 720)), use_rgb=True)
        
        Image.fromarray(cam_image).save("{}-cam_yolop-{}.jpeg".format(str(opt.save_dir +'/'+ Path(path).name), opt.layer+1))
        
    print('Results saved to %s' % Path(opt.save_dir))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=10, help='layer in the model')
    parser.add_argument('--task', type=str, default="ll_seg", help='target task (det/da_seg/ll_seg)')
    parser.add_argument('--label', type=bool, default=True, help='whether to show labels on picture')
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
