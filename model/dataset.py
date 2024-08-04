import os.path
import pdb
import torchvision.transforms as transforms
import os.path as osp

from PIL import Image
import PIL
import numpy as np
import torch
from torch.autograd import Variable
import cv2
pwd = osp.split(osp.realpath(__file__))[0]
import sys
sys.path.append(pwd + '/..')
import faceutils as futils
import torch.nn.functional as F


def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

def to_var(x, requires_grad=True):
    if requires_grad:
        return Variable(x).float()
    else:
        return Variable(x, requires_grad=requires_grad).float()


class InferenceDataset():
    def __init__(self, device, makeup_paths, non_makeup_paths):
        self.random = None
        self.n_componets = 3
        self.makeup_paths = makeup_paths
        self.non_makeup_paths = non_makeup_paths
        self.device = device
        
        self.image_size = 256
        self.face_parse = futils.mask.FaceParser(device=self.device)
        self.up_ratio = 0.6 / 0.85
        self.down_ratio = 0.2 / 0.85
        self.width_ratio = 0.2 / 0.85
        self.img_size = 256
        self.lip_class   = [7,9]
        self.face_class  = [1,6]
        self.eyes_class = [4,5]

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.transform_mask = transforms.Compose([
            # transforms.Resize((img_size, img_size), interpolation=PIL.Image.NEAREST),
            ToTensor])
        
    def __len__(self):
        return len(self.makeup_paths)
    
    def __getitem__(self, idx):
        makeup_path = self.makeup_paths[idx]
        non_makeup_path = self.non_makeup_paths[idx]
        
        makeup_image = Image.open(makeup_path).convert('RGB')
        non_makeup_image = Image.open(non_makeup_path).convert('RGB')
        
        face = futils.dlib.detect(makeup_image)
        try:
            face_on_image = face[0]
        except:
            return {}


        image_makeup, face, crop_face = futils.dlib.crop(makeup_image, face_on_image, self.up_ratio, self.down_ratio, self.width_ratio)
        np_image_makeup = np.array(image_makeup)
        # pdb.set_trace()
        mask = self.face_parse.parse(cv2.resize(np_image_makeup, (512, 512)))
        mask = F.interpolate(mask.view(1, 1, 512, 512), (self.img_size, self.img_size), mode="nearest")
        mask = mask.type(torch.uint8)
        mask = to_var(mask, requires_grad=False).to(self.device)

        mask_B_lip = (mask == self.lip_class[0]).float() + (mask == self.lip_class[1]).float()
        mask_B_face = (mask == self.face_class[0]).float() + (mask == self.face_class[1]).float()
        mask_B_eye_left = (mask == self.eyes_class[0]).float()
        mask_B_eye_right= (mask == self.eyes_class[1]).float()

        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left[0], mask_B_eye_right[0], mask_B_face[0])

        mask_eyes = mask_B_eye_left + mask_B_eye_right
        mask_list = [mask_B_lip, mask_B_face, mask_eyes]
        makeup_seg = torch.cat(mask_list, 0) 
        makeup_seg = makeup_seg[:,0,:,:]
        makeup_unchanged = (mask == 0).float().squeeze(0)


        nonmakeface = futils.dlib.detect(non_makeup_image)
        try:
            face_on_image = nonmakeface[0]  
        except:
            return{} 
        image_nonmakeup, face, crop_face = futils.dlib.crop(
        non_makeup_image, face_on_image, self.up_ratio, self.down_ratio, self.width_ratio)
        np_image_nomakeup = np.array(image_nonmakeup)
        mask = self.face_parse.parse(cv2.resize(np_image_nomakeup, (512, 512)))

        mask = F.interpolate(
            mask.view(1, 1, 512, 512),
            (self.img_size, self.img_size),
            mode="nearest")
        mask = mask.type(torch.uint8)
        mask = to_var(mask, requires_grad=False).to(self.device)

        
        mask_A_lip = (mask == self.lip_class[0]).float() + (mask == self.lip_class[1]).float()
        mask_A_face = (mask == self.face_class[0]).float() + (mask == self.face_class[1]).float()
        mask_A_eye_left = (mask == self.eyes_class[0]).float()
        mask_A_eye_right= (mask == self.eyes_class[1]).float()


        # if not ((mask_A_eye_left > 0).any() and \
        #         (mask_A_eye_right > 0).any()):
        #     return {}
    
        # mask_eyes = (mask == self.eyes_class[0]).float() + (mask == self.eyes_class[1]).float()
        mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left[0], mask_A_eye_right[0], mask_A_face[0])
        # if (mask_A_eye_left + mask_A_eye_right).max()>1.5:
        #     # print('error')
        #     return {}

        mask_eyes = mask_A_eye_left + mask_A_eye_right
        mask_list = [mask_A_lip, mask_A_face, mask_eyes]
        nonmakeup_seg = torch.cat(mask_list, 0) 
        nonmakeup_seg = nonmakeup_seg[:,0,:,:]

        nonmakeup_unchanged = (mask == 0).float().squeeze(0)

        mask_A_face, mask_B_face, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_face, mask_B_face) 
        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)

        mask_A = {}
        mask_A["mask_A_eye_left"] = mask_A_eye_left
        mask_A["mask_A_eye_right"] = mask_A_eye_right
        mask_A["index_A_eye_left"] = index_A_eye_left
        mask_A["index_A_eye_right"] = index_A_eye_right
        mask_A["mask_A_skin"] = mask_A_face
        mask_A["index_A_skin"] = index_A_skin
        mask_A["mask_A_lip"] = mask_A_lip
        mask_A["index_A_lip"] = index_A_lip

        mask_B = {}
        mask_B["mask_B_eye_left"] = mask_B_eye_left
        mask_B["mask_B_eye_right"] = mask_B_eye_right
        mask_B["index_B_eye_left"] = index_B_eye_left
        mask_B["index_B_eye_right"] = index_B_eye_right
        mask_B["mask_B_skin"] = mask_B_face
        mask_B["index_B_skin"] = index_B_skin
        mask_B["mask_B_lip"] = mask_B_lip
        mask_B["index_B_lip"] = index_B_lip

        makeup_img = self.transform(image_makeup)
        nonmakeup_img = self.transform(image_nonmakeup)
        return {
            'nonmakeup_seg': nonmakeup_seg,
            'makeup_seg': makeup_seg, 
            'nonmakeup_img': nonmakeup_img,
            'makeup_img': makeup_img,
            'mask_A': mask_A, 
            'mask_B': mask_B,
            'makeup_unchanged': makeup_unchanged,
            'nonmakeup_unchanged': nonmakeup_unchanged,
            'nonmakeup_name':non_makeup_path.split('/')[-1],
            'makeup_name':makeup_path.split('/')[-1]
        }


    def mask_preprocess(self, mask_A, mask_B):
        # pdb.set_trace()
        # mask_A = mask_A.unsqueeze(0)
        # mask_B = mask_B.unsqueeze(0)
        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]

        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        return mask_A, mask_B, index, index_2

def test():
    from torch.utils.data import DataLoader
    test_dataset = InferenceDataset(device='cuda', makeup_paths=['./data/makeup/1.jpg'], non_makeup_paths=['./data/nonmakeup/1.jpg'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for i, data in enumerate(test_loader):
        print(data)
        break 

if __name__ == '__main__': test()