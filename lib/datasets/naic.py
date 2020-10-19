# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import random
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils import data
import albumentations as A
import albumentations.augmentations.functional as Fa
from lib.datasets import uniform
import glob
# np.set_printoptions(threshold=np.inf)
align_corners = True
class NAIC(data.Dataset):
    def __init__(self, 
                 data_path,
                 num_classes=8,
                 num_samples=None,
                 ignore_label=-1, 
                 base_size=256, 
                 crop_size=(256, 256), 
                 downsample_rate=1,
                 multi_scale=False,
                 flip=False, 
                 scale_factor=100,
                 mean=[0.355, 0.383, 0.358], 
                 std=[0.207, 0.201, 0.210]):
                 
        self.base_size = base_size
        self.crop_size = crop_size#(h,w)
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.centroids = None
        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1./downsample_rate
        self.multi_scale = multi_scale
        self.flip = flip
        self.extra_datapath ='/workspace/13_raid/xuzekun/naic_remote_scence/naic_dataset/val'
        self.images_extra_datapath = os.path.join( self.extra_datapath, 'image')
        self.labels_extra_datapath = os.path.join( self.extra_datapath, 'label')
        self.data_path = data_path
        self.images_path = os.path.join( self.data_path, 'image')
        self.labels_path = os.path.join( self.data_path, 'label')
        if 'train' in self.data_path:
            self.train = True
        else:
            self.train = False
        img_ext = 'tif'
        mask_ext = 'png'
        if 'test' in self.data_path:
            self.all_imgs = self.make_dataset_folder(self.data_path)
            self.build_epoch()
        elif 'train' in self.data_path:
            self.all_imgs = self.find_images(self.images_path, self.labels_path, img_ext,
                                             mask_ext)#[(image_fn, label_fn)]
            self.all_imgs.extend(
                self.find_images(self.images_extra_datapath, self.labels_extra_datapath, img_ext, mask_ext)
                                             )
            self.centroids = uniform.build_centroids(self.all_imgs,
                                                    self.num_classes,
                                                    self.train,#bool
                                                    )
            self.build_epoch()
        else:
            self.all_imgs = self.find_images(self.images_path, self.labels_path, img_ext,
                                             mask_ext)
                                             
            self.centroids = uniform.build_centroids(self.all_imgs,
                                                    self.num_classes,
                                                    self.train,#bool
                                                    )
            self.build_epoch()
        if self.num_samples:
            self.images_list = self.images_list[:self.num_samples+1]

        # self.files = self.read_files()

        self.class_weights = torch.FloatTensor([1.0]*8).cuda()


    def build_epoch(self):
        """
        For class uniform sampling ... every epoch, we want to recompute
        which tiles from which images we want to sample from, so that the
        sampling is uniformly random.
        """
        self.imgs = uniform.build_epoch(self.all_imgs,
                                        self.centroids,
                                        self.num_classes,
                                        self.train)#list,[[image_fn, label_fn]/[image_fn, label_fn, centroid,class_id]]
    
    @staticmethod
    def find_images(img_root, mask_root, img_ext, mask_ext):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.
        """
        img_path = '{}/*.{}'.format(img_root, img_ext)
        imgs = glob.glob(img_path)
        items = []
        for full_img_fn in imgs:
            img_dir, img_fn = os.path.split(full_img_fn)
            img_name, _ = os.path.splitext(img_fn)
            full_mask_fn = '{}.{}'.format(img_name, mask_ext)
            full_mask_fn = os.path.join(mask_root, full_mask_fn)
            assert os.path.exists(full_mask_fn)
            items.append((full_img_fn, full_mask_fn))
        return items

    @staticmethod
    def make_dataset_folder(folder):
        """
        Create Filename list for images in the provided path

        input: path to directory with *only* images files
        returns: items list with None filled for mask path
        """
        items = os.listdir(folder)
        items = [(os.path.join(folder, f), '') for f in items]
        items = sorted(items)

        print(f'Found {len(items)} folder imgs')

        """
        orig_len = len(items)
        rem = orig_len % 8
        if rem != 0:
            items = items[:-rem]

        msg = 'Found {} folder imgs but altered to {} to be modulo-8'
        msg = msg.format(orig_len, len(items))
        print(msg)
        """

        return items

    def __len__(self):
        return len(self.all_imgs)
    
    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]#BGR2RGB
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image
    
    def label_transform(self, label):
        label = label//100 -1
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=padvalue)
        
        return pad_image

    def rand_crop(self, image, label, centroid=None):
        # h, w = image.shape[:-1]
        # image = self.pad_image(image, h, w, self.crop_size,
        #                         (0.0, 0.0, 0.0))
        # label = self.pad_image(label, h, w, self.crop_size,
        #                         (self.ignore_label,))
        
        # new_h, new_w = label.shape
        # x = random.randint(0, new_w - self.crop_size[1])
        # y = random.randint(0, new_h - self.crop_size[0])
        # image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        # label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        # return image, label
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                                (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                                (self.ignore_label,))
        new_h, new_w = label.shape
        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            # print('cx,cy',c_x,c_y)
            max_x = new_w - self.crop_size[1]
            max_y = new_h - self.crop_size[0]
            x1 = random.randint(c_x - self.crop_size[1], c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - self.crop_size[0], c_y)
            y1 = min(max_y, max(0, y1))
            image = image[y1:y1+self.crop_size[0],x1:x1+self.crop_size[1]]
            label = label[y1:y1+self.crop_size[0],x1:x1+self.crop_size[1]]
            # print("image_centroid.shape",image.shape)
            # print('label_centroid.shape',label.shape)
        else:
            x = random.randint(0, new_w - self.crop_size[1])
            y = random.randint(0, new_h - self.crop_size[0])
            image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
            label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
            # print("image.shape",image.shape)
            # print('label.shape',label.shape)
        return image, label

    def multi_scale_aug(self, image, label=None, 
            rand_scale=1, rand_crop=True,centroid=None):
        if centroid is not None:
            centroid = [int(c * rand_scale) for c in centroid]
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        
        image = cv2.resize(image, (new_w, new_h), 
                           interpolation = cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), 
                           interpolation = cv2.INTER_NEAREST)
        else:
            return image
        
        if rand_crop:
            image, label = self.rand_crop(image, label,centroid)
        
        return image, label

    def gen_sample(self, image, label, 
            multi_scale=True, is_flip=True, centroid=None):
        if multi_scale:
            rand_scale = 0.3 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, 
                                                rand_scale=rand_scale,centroid=centroid)

        image = self.input_transform(image)
        label = self.label_transform(label)

        train_trasform = A.Compose(
            [
                A.RandomRotate90(p=0.1),
                A.Transpose(p=0.1),
                A.HorizontalFlip(p=0.1),
                A.VerticalFlip(p=0.1)
            ]
        )
        transformed = train_trasform(image = image,mask = label)
        image = transformed['image']
        label = transformed['mask']
        
        image = image.transpose((2, 0, 1))#HWC2CHW
        
        # if is_flip:
        #     flip = np.random.choice(2) * 2 - 1
        #     image = image[:, :, ::flip]
        #     label = label[:, ::flip]

        #     flip = np.random.choice(2) * 2 - 1
        #     image = image[:, ::flip, :]
        #     label = label[::flip, :]

        if self.downsample_rate != 1:
            label = cv2.resize(label, 
                               None, 
                               fx=self.downsample_rate,
                               fy=self.downsample_rate, 
                               interpolation=cv2.INTER_NEAREST)

        return image, label

    def inference(self, model, image, config,flip=False):
        size = image.size()
        pred = model(image)
        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        pred = F.interpolate(input=pred, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear',align_corners=True)        
        if flip:
            flip_img = image.numpy()[:,:,:,::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]
            flip_output = F.interpolate(input=flip_output, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear',align_corners=True)
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()
#-----------------------------------------------------------
    # def multi_scale_inference(self, model, image, config,scales=[1], flip=False):
    #     batch, _, ori_height, ori_width = image.size()
    #     assert batch == 1, "only supporting batchsize 1."
    #     image = image.numpy()[0].transpose((1,2,0)).copy()
    #     stride_h = np.int(self.crop_size[0] * 2.0 / 3.0)
    #     stride_w = np.int(self.crop_size[1] * 2.0 / 3.0)
    #     final_pred = torch.zeros([1, self.num_classes,
    #                                 ori_height,ori_width]).cuda()
    #     padvalue = -1.0  * np.array(self.mean) / np.array(self.std)
    #     for scale in scales:
    #         new_img = self.multi_scale_aug(image=image,
    #                                        rand_scale=scale,
    #                                        rand_crop=False)
    #         height, width = new_img.shape[:-1]
                
    #         if max(height, width) <= np.min(self.crop_size):
    #             new_img = self.pad_image(new_img, height, width, 
    #                                 self.crop_size, padvalue)
    #             new_img = new_img.transpose((2, 0, 1))
    #             new_img = np.expand_dims(new_img, axis=0)
    #             new_img = torch.from_numpy(new_img)
    #             preds = self.inference(model, new_img, config,flip)
    #             preds = preds[:, :, 0:height, 0:width]
    #         else:
    #             if height < self.crop_size[0] or width < self.crop_size[1]:
    #                 new_img = self.pad_image(new_img, height, width, 
    #                                     self.crop_size, padvalue)
    #             new_h, new_w = new_img.shape[:-1]
    #             rows = np.int(np.ceil(1.0 * (new_h - 
    #                             self.crop_size[0]) / stride_h)) + 1
    #             cols = np.int(np.ceil(1.0 * (new_w - 
    #                             self.crop_size[1]) / stride_w)) + 1
    #             preds = torch.zeros([1, self.num_classes,
    #                                        new_h,new_w]).cuda()
    #             count = torch.zeros([1,1, new_h, new_w]).cuda()

    #             for r in range(rows):
    #                 for c in range(cols):
    #                     h0 = r * stride_h
    #                     w0 = c * stride_w
    #                     h1 = min(h0 + self.crop_size[0], new_h)
    #                     w1 = min(w0 + self.crop_size[1], new_w)
    #                     crop_img = new_img[h0:h1, w0:w1, :]
    #                     if h1 == new_h or w1 == new_w:
    #                         crop_img = self.pad_image(crop_img, 
    #                                                   h1-h0, 
    #                                                   w1-w0, 
    #                                                   self.crop_size, 
    #                                                   padvalue)
    #                     crop_img = crop_img.transpose((2, 0, 1))
    #                     crop_img = np.expand_dims(crop_img, axis=0)
    #                     crop_img = torch.from_numpy(crop_img)
    #                     pred = self.inference(model, crop_img, config,flip)
    #                     preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
    #                     count[:,:,h0:h1,w0:w1] += 1
    #             preds = preds / count
    #             preds = preds[:,:,:height,:width]
    #         preds = F.upsample(preds, (ori_height, ori_width), 
    #                                mode='bilinear')
    #         final_pred += preds
    #     return final_pred
#-----------------------------------------------------------
    def multi_scale_inference(self, model, image, config,scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        
        final_pred = torch.zeros([batch, self.num_classes,
                                     ori_height,ori_width]).cuda()
        for scale in scales:
            # new_img = self.multi_scale_aug(image=image,
            #                                rand_scale=scale,
            #                                rand_crop=False)
            new_img = F.interpolate(image, scale_factor=scale, 
                                            mode='bilinear', 
                                            align_corners=True)

            preds = self.inference(model, new_img, config,flip)

            preds = F.interpolate(preds, (ori_height, ori_width), 
                                   mode='bilinear',align_corners=True)
            final_pred += preds
        return final_pred

    def read_files(self):
        files = []
        if 'test' in self.data_path:
            for item in self.images_list:
                name = os.path.splitext(os.path.basename(item))[0]
                files.append({
                    "img_path": item,
                    "name": name,
                })
        elif 'train' in self.data_path:
            # for image_path in self.images_list:
            #     name = os.path.splitext(os.path.basename(image_path))[0]
            #     files.append({
            #         "img_path": image_path,
            #         "label_path": os.path.join(self.labels_path,name+'.png'),
            #         "name": name,
            #         "weight": 1
            #     })
            # for image_path in self.images_extra_list:
            #     name = os.path.splitext(os.path.basename(image_path))[0]
            #     files.append({
            #         "img_path": image_path,
            #         "label_path": os.path.join(self.labels_extra_datapath,name+'.png'),
            #         "name": name,
            #         "weight": 1
            #     })
            for img_and_mask_path in self.all_imgs:
                image_path = img_and_mask_path[0]
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img_path": image_path,
                    "label_path":img_and_mask_path[1],
                    "name": name,
                    "weight": 1
                })
        else:
            # for image_path in self.images_list:
            #     name = os.path.splitext(os.path.basename(image_path))[0]
            #     files.append({
            #         "img_path": image_path,
            #         "label_path": os.path.join(self.labels_path,name+'.png'),
            #         "name": name,
            #         "weight": 1
            #     })

            for img_and_mask_path in self.all_imgs:
                image_path = img_and_mask_path[0]
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img_path": image_path,
                    "label_path":img_and_mask_path[1],
                    "name": name,
                    "weight": 1
                })

        return files
#---------------------------
    # def __getitem__(self,index):
    #     item = self.files[index]
    #     name = item["name"]
    #     image = cv2.imread(item["img_path"],
    #                        cv2.IMREAD_COLOR)#hwc
    #     size = image.shape

    #     if 'test' in self.data_path:
    #         image = self.input_transform(image)
    #         image = image.transpose((2, 0, 1))

    #         return image.copy(), np.array(size), name

    #     label = cv2.imread(item["label_path"],cv2.IMREAD_ANYDEPTH)

    #     image, label = self.gen_sample(image, label, 
    #                             self.multi_scale, self.flip)

    #     return image.copy(), label.copy(), np.array(size), name
#--------------------------------
    def __getitem__(self,index):

        if len(self.imgs[index]) == 2:#test
            img_path, mask_path = self.imgs[index]
            centroid = None
            class_id = None
        else:
            img_path, mask_path, centroid, class_id = self.imgs[index]

        name = os.path.splitext(os.path.basename(img_path))[0]

        image = cv2.imread(img_path,
                           cv2.IMREAD_COLOR)#hwc
        size = image.shape

        if 'test' in self.data_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(mask_path,cv2.IMREAD_ANYDEPTH)

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip,centroid)

        return image.copy(), label.copy(), np.array(size), name
    #---------------------------
    def get_palette(self,n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette
    
    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu()
        # print(preds[1])
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint16)+1
        # print(preds[1])
        preds = preds*100
        # print(preds[1])
        for i in range(preds.shape[0]):
            # print(type(preds[i]))
            # print(preds[i].shape)
            # print(preds[i])
            # print(preds[i])
            save_img = Image.fromarray(preds[i])
            # save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
        
            # sys.exit()


if __name__ == "__main__":
    train_dataset = NAIC("/workspace/13_raid/xuzekun/naic_remote_scence/train")
    for i in range(1):
        print(train_dataset[i][1].shape)
        print("-------------------------------------")
