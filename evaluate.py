import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from PIL import Image
from numpy.lib.function_base import rot90
from misc import check_mkdir, cal_precision_recall_mae, AvgMeter, cal_fmeasure, cal_Jaccard, cal_BER
from tqdm import tqdm
import argparse
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, default='/models', help='model name')
parser.add_argument('--snapshot', type=str, default='best', help='model name')
tmp_args = parser.parse_args()

root_path = f'./ckpt/{tmp_args.models}/predict_{tmp_args.snapshot}'
save_path = f'./ckpt/{tmp_args.models}/predict_{tmp_args.snapshot}'
fail_path = f'./ckpt/{tmp_args.models}/predict_fail_{tmp_args.snapshot}'

f_path = os.path.abspath('..')
# print(f_path.split('shadow_code'))
ll_path = f_path.split('shadow_code')[0]
# print(root_path)
gt_path = ll_path+'datasets/shadow/test/labels'
input_path = ll_path+'datasets/shadow/test/images'
resize_transform = transforms.Compose([
    transforms.Resize((320,320)),
])
precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
mae_record = AvgMeter()
Jaccard_record = AvgMeter()
BER_record = AvgMeter()
shadow_BER_record = AvgMeter()
non_shadow_BER_record = AvgMeter()

video_list = os.listdir(root_path)
for video in tqdm(video_list):
    gt_list = os.listdir(os.path.join(gt_path, video))
    
    # img_list = [f for f in os.listdir(os.path.join(root_path, video)) if f.split('_', 1)[0]+'.png' in gt_list]  # include overlap images
    img_list = [f for f in os.listdir(os.path.join(root_path, video)) if f in gt_list]  
    
    # include overlap images
    img_set = list(set([img.split('_', 1)[0] for img in img_list]))  # remove repeat
    for img_prefix in img_set:
        # jump exist images
        check_mkdir(os.path.join(save_path, video))
       
        # save_name = os.path.join(save_path, video, '{}.png'.format(img_prefix))
        save_name = os.path.join(save_path, video, '{}'.format(img_prefix))
        # if not os.path.exists(os.path.join(save_path, video, save_name)):
        # imgs = [img for img in img_list if img.split('_', 1)[0] == img_prefix]  # imgs waited for fuse
        # fuse = []
        # for img_path in imgs:
        #     img = np.array(Image.open(os.path.join(root_path, video, img_path)).convert('L')).astype(np.float32)
        #     # if np.max(img) > 0:  # normalize prediction mask
        #     #     img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        #     # print(img.shape)
        #     fuse.append(img)
        # print(len(imgs))

        # print(sum(fuse).shape)
        # fuse = (sum(fuse) / len(imgs)).astype(np.uint8)
        # save image
        # print(f'Save:{save_name}')
        # Image.fromarray(fuse).save(save_name)
        # else:
        # print(f'Exist:{save_name}')
        ori_fuse =np.array(Image.open(save_name).convert('L'))
        # print(save_name)
        fuse = np.array(Image.open(save_name).convert('L')).astype(np.uint8)
        # calculate metric
        gt = Image.open(os.path.join(gt_path, video, img_prefix))
        # gt=resize_transform(gt)
        # gt=resize_transform(gt)
        gt = np.array(gt)
        # print(gt.shape, fuse.shape)
        # ssss
        # print()
        
        # for k in range(0,256):
        #     for i in range( gt.shape[0]):
        #         for j in range(gt.shape[1]):
        #             if gt[i][j]==k:
        #                 print(k)

        precision, recall, mae = cal_precision_recall_mae(fuse, gt)
        # np.where(gt>254,255,0)
        # np.where(fuse>254,255,0)
        # print(fuse.shape, gt.shape)
        Jaccard = cal_Jaccard(fuse, gt)
        Jaccard_record.update(Jaccard)
       
        BER, shadow_BER, non_shadow_BER = cal_BER(fuse/255., gt/255.)
        BER_record.update(BER)
        shadow_BER_record.update(shadow_BER)
        non_shadow_BER_record.update(non_shadow_BER)
        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            precision_record[pidx].update(p)
            recall_record[pidx].update(r)
        
        # print(mae)
        fuse = fuse / 255.
        gt = gt / 255.
        mae = np.mean(np.abs(fuse - gt))
        mae_record.update(mae)
        # print(mae)
        if mae>0.03:
            check_mkdir(os.path.join(fail_path, video))
            fail_name = os.path.join(fail_path, video, '{}.png'.format(img_prefix))
            # print(mae)

            Image.fromarray(ori_fuse).save(fail_name)
       
# print("MAE:{}, BER:{}, shadow_BER:{}, non_shadow_BER:{}, Jaccard: {} ".format(mae_record.avg, BER_record.avg, shadow_BER_record.avg,non_shadow_BER_record.avg))
fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                        [rrecord.avg for rrecord in recall_record])
# fmeasure =0.0
log = 'MAE:{}, F-beta:{}, Jaccard:{}, BER:{}, SBER:{}, non-SBER:{}'.format(mae_record.avg, fmeasure, Jaccard_record.avg, BER_record.avg, shadow_BER_record.avg, non_shadow_BER_record.avg)
print(log)


