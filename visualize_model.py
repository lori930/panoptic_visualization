import torch
from torchviz import make_dot
import numpy as np
import cv2 
from PIL import Image

# Need YOLOP installed: git clone git@github.com:hustvl/YOLOP.git
from lib.models import get_net
from lib.config import cfg
import torchvision.transforms as transforms

model = get_net(cfg)
checkpoint = torch.load("weights/End-to-end.pth", map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.cpu()


image_url = "https://upload.wikimedia.org/wikipedia/commons/f/f1/Puppies_%284984818141%29.jpg"
img = np.array(Image.open(requests.get(image_url, stream=True).raw))
img = cv2.resize(img, (640, 640))
# print(img.shape)
rgb_img = img.copy()
# img = transform(img)
img = np.float32(img) / 255  # uint8 to fp16/32
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)
output = model(tensor)

make_dot(output[0][0], params=dict(model.named_parameters())).render("yolop_torchviz-simplified-det.pdf")
