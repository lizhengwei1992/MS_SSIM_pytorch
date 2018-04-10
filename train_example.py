import loss
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np

npImg1 = cv2.imread("einstein.png")

img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
img2 = torch.rand(img1.size())

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()


img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2, requires_grad = True)


# according input set max_val : 255 or 1
ms_ssim_loss = MS_SSIM(max_val = 1)

optimizer = optim.Adam([img2], lr=0.01)

while ssim_value < 0.97:
    optimizer.zero_grad()
    ms_ssim_out = -ms_ssim_loss(img1, img2)
    ms_ssim_value = - ms_ssim_out.data[0]
    print(ms_ssim_value)
    ms_ssim_out.backward()
    optimizer.step()
