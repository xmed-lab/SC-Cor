import os
import os.path
import torch
import torch.utils.data as data
from PIL import Image
from PIL import ImageEnhance
import random
import numpy as np

def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'images')) if f.endswith('.jpg')]
    print(img_list)
    return [
        (os.path.join(root, 'images', img_name + '.jpg'), os.path.join(root, 'ShadowMasks', img_name + '.png'),\
        os.path.join(root, 'labels', img_name + '.png'),os.path.join(root, 'fuse_dst2', img_name + '.png'))
        for img_name in img_list]


class ImageFolder(data.Dataset):
    def __init__(self, args,root, joint_transform=None, transform=None, target_transform=None,training=False):
        self.root = root
        # self.imgs = make_dataset(root)
        self.args = args
        self.training = training
        self.img_root, self.video_root = self.split_root(root)
        # print(self.video_root)
        self.input_folder = 'images'
        self.label_folder = 'labels'
        self.img_ext = '.jpg'
        self.label_ext = '.png'
        self.num_video_frame = 0
        self.imgs = self.generateImgFromVideo(self.video_root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
       
    def __getitem__(self, index):
        img_path, gt_path, img_name, c_index, videoLength = self.imgs[index]
        s,e =self.args['s'],self.args['e']
        # print(s,e)
        bright_random = random.uniform(s,e) 
        
        # dst1 = Image.open(dst1_path).convert('RGB')
        # dst2 = Image.open(dst2_path).convert('RGB')
        if self.training:
            sample_len = self.args['sample']
            
            start_index = max(1,c_index-sample_len)
            end_index = min(videoLength,c_index+sample_len)
            if random.random()>0.5:
                query_index = np.random.randint(start_index,  end_index)
                query_index = start_index
            else:
                query_index = end_index
            # while query_index==c_index:
            #     query_index = np.random.randint(start_index,  end_index)
        else:
            # if c_index == videoLength:
            #     query_index=c_index-1
            # else:
                query_index = c_index+1
        if c_index == 1:
            # query_index = 1
            e_query_index = np.random.randint(c_index+1,  videoLength+1)
        elif c_index==videoLength:
            # query_index = np.random.randint(1,  c_index)
            e_query_index=videoLength
        else:
            # query_index = np.random.randint(1,  c_index)
            e_query_index = np.random.randint(c_index+1,  videoLength+1)
        # while c_index==query_index:
        #     query_index = np.random.randint(start_index,  end_index)
        
        query_path = img_path.replace(img_name,'%08d' % query_index)
        query_gt_path = gt_path.replace(img_name,'%08d' % query_index)

        e_query_path = img_path.replace(img_name,'%08d' % e_query_index)
        e_query_gt_path = gt_path.replace(img_name,'%08d' % e_query_index)

        img = Image.open(img_path).convert('RGB')
        # print(gt_path)
        target = Image.open(gt_path).convert('L')

        query_img = Image.open(query_path).convert('RGB')
        # print(gt_path)
        query_target = Image.open(query_gt_path).convert('L')

        # e_query_img = Image.open(e_query_path).convert('RGB')
        # e_query_target = Image.open(e_query_gt_path).convert('L')
        # print(img_path,query_path)
        # sss
        output=dict()
        if self.joint_transform is not None:
            # img, target = self.joint_transform(img, target)
            # query_img, query_target = self.joint_transform(query_img, query_target)
            img, target, query_img, query_target = self.joint_transform(img, target, query_img, query_target)

            if self.args['brightness']:
                # print(self.training)
                # print(bright_random)
                e_query_img = query_img.copy()
                query_img = query_img.point(lambda p: p * bright_random)
            if self.args['e_query']:
                output['e_query_img'] = e_query_img
                # img_b = img.point(lambda p: p * bright_random)
                # output['img_b'] = img_b
                # output['query_img_b'] = query_img_b
                # print(e_query_img==query_img)
            output['img'] = img
            output['target'] = target
            output['query_img'] = query_img
            output['query_target'] = query_target
           
            # output['e_query_target'] = e_query_target

        if self.transform is not None:

            # img = random.uniform(1.0,1.2)*img
            # img = torch.clamp(img, 0, 255)
            # enh_bri = ImageEnhance.Brightness(img)
            # img = enh_bri.enhance(random.uniform(1.0,1.2))
            # img = torch.clamp(img, 0, 255)
            img = self.transform(img)
            query_img = self.transform(query_img)
            
            output['img'] = img
            # output['target'] = target
            output['query_img'] = query_img
            if self.args['e_query']:
                e_query_img = self.transform(e_query_img)
                output['e_query_img'] = e_query_img
            # output['query_target'] = query_target
            # if self.args['brightness']:
            #     output['img_b'] = self.transform(img_b)
            #     output['query_img_b'] = self.transform(query_img_b)

        if self.target_transform is not None:
            target = self.target_transform(target)
            query_target = self.target_transform(query_target)
            # e_query_target = self.target_transform(e_query_target)
            output['target'] = target
            output['query_target'] = query_target
            
            # output['e_query_target'] = e_query_target
            # dst1 = self.target_transform(dst1)
            # dst2 = self.target_transform(dst2)
        # print(img.shape,target.shape)
        output['img_path']= img_path
        output['query_path'] = query_path
        output['gt_path']= gt_path
        output['query_gt_path'] = query_gt_path
        # return img, target
        return output

    def __len__(self):
        return len(self.imgs)

    def generateImgFromVideo(self, root):
        imgs = []
        root = root[0]  # assume that only one video dataset
        video_list = os.listdir(os.path.join(root[0].replace('fixedshadow','shadow'), self.input_folder))
        for video in video_list:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root[0].replace('fixedshadow','shadow'), self.input_folder, video)) if f.endswith(self.img_ext)] # no ext
            img_list = self.sortImg(img_list)
            start_index = 0 
            length = len(img_list)
            for index, img in enumerate(img_list):
                # videoImgGt: (img, gt, video start index, video length)
                # videoImgGt = (os.path.join(root[0], self.input_folder, video, img + self.img_ext),
                #         os.path.join(root[0], self.label_folder, video, img + self.label_ext), self.num_video_frame, len(img_list))
                if  not self.training:
                    if index == length:
                        continue
                videoImgGt = [os.path.join(root[0].replace('fixedshadow','shadow'), self.input_folder, video, img + self.img_ext),
                        os.path.join(root[0], self.label_folder, video, img + self.label_ext),
                        img,
                        index+1,
                        length
                  # os.path.join(root[0], self.label_folder, video+'_FN', img + self.label_ext),
                        #  os.path.join(root[0], self.label_folder, video+'_FP', img + self.label_ext),
                        ]
                imgs.append(videoImgGt)
            self.num_video_frame += len(img_list)
        return imgs

    def split_root(self, root):
        if not isinstance(root, list):
            raise TypeError('root should be a list')
        img_root_list = []
        video_root_list = []
        for tmp in root:
            if tmp[1] == 'image':
                
                img_root_list.append(tmp)
            elif tmp[1] == 'video':
                video_root_list.append(tmp)
            else:
                raise TypeError('you should input video or image')
        return img_root_list, video_root_list

    def sortImg(self, img_list):
        img_int_list = [int(f) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]