import numpy as np
import os
gpustr = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = gpustr
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from config import ViSha_validation_root, ViSha_training_root
from misc import check_mkdir, crf_refine
from model import DSDNet
import sys
torch.cuda.set_device(0)
import random
ckpt_path = './ckpt'

# exp_name='ViSha_corres_2_sample_1_(non_bright_0.7,1.3)_10_P:False_grid17/'
exp_name='models/'
args = {
    # 'snapshot': 'best_mae_2834',
    'snapshot': 'best',
    'scale': 320,
     'input_folder': 'images',
    'label_folder': 'labels',
    'fixedBN': True,
    'freeze_bn_affine':True,
    'grid':17
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'],args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# to_test = {'sbu': ViSha_validation_root}
# root = ViSha_img_training_root[0]
root = ViSha_validation_root[0]
to_pil = transforms.ToPILImage()

bright_random = random.uniform(1.0,2.0) 

def main():
    net = DSDNet(args)
    net = torch.nn.DataParallel(net).cuda()
    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    net.eval()
    with torch.no_grad():
        video_list = os.listdir(os.path.join(root.replace('fixedshadow','shadow'), args['input_folder']))
        video_list = video_list[0:70]
        # video_list = ["Car1"]
        for video in tqdm(video_list):
            # print(video)
            # video = "Car1"
            img_list = [f for f in os.listdir(os.path.join(root.replace('fixedshadow','shadow'), args['input_folder'], video)) if
                        f.endswith('.jpg')]
            # os.path.splitext(f)[0]
            for idx, img_name in enumerate(img_list):
                print('predicting for %s: %d / %d' % (video, idx + 1, len(img_list)))
               
                img = Image.open(os.path.join(root.replace('fixedshadow','shadow'), args['input_folder'], video,img_name)).convert('RGB')
                w, h = img.size
                # w, h = 320, 320
                img_list = []
                img_var = Variable(img_transform(img)).unsqueeze(0).cuda()
        
                inputs = dict()
                inputs['img']=img_var
                res = net(inputs)
                res = (res.data > 0).to(torch.float32)
                prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
                # prediction = transforms.Resize((h, w))(img.convert('RGB'))
                # prediction = crf_refine(np.array(r_img), prediction)
                img_name = img_name.split('.')[0]
                check_mkdir(os.path.join(ckpt_path, exp_name, "predict_" + args['snapshot'], video))
                Image.fromarray(prediction).save(
                        os.path.join(ckpt_path, exp_name, "predict_" + args['snapshot'], video, img_name+'.png'))
                
      

if __name__ == '__main__':
    main()
