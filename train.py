import datetime
import os
gpustr = "1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = gpustr
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import joint_transforms
from config import ViSha_validation_root,ViSha_training_root
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir, cal_BER, cal_temporal, cal_temporal2
from model import DSDNet
from utils import MyBceloss12_n,MyWcploss
import numpy as np
from tqdm import tqdm
cudnn.benchmark = True

# torch.cuda.set_device(0)

ckpt_path = './ckpt'

# exp_name=sys.argv[1]
from losses import lovasz_hinge, binary_xloss, pc_loss, margin_loss
args = {
    'iter_num': 5000,
    'train_batch_size': 22,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 1e-3,
    'momentum': 0.9,
    'snapshot': '5000',
    'scale':320,
    'dist':False,
    'brightness': True,
    'weight': 10,
    's':0.7,
    'e':1.3,
    'pre_train':False,
    # 'two':True,
    'sample':1,
    'e_query': False,
    'consis': False,
    'non_shadow':False,
    'fixedBN': False,
    'freeze_bn_affine':False,
    'pre_con':False,
    'free_iter':5000,
    'grid':17,
    'beta':0.3
    
    
}
if args['brightness']:
    q='bright'
else:
    q='non_bright'
exp_name = 'ViSha_corres_2'+'_sample_{}_({}_{},{})_{}_P:{}'.format(args['sample'],q,args['s'],args['e'],args['weight'],args['pre_train'])
if args['consis']:
    exp_name+="_consis"
if args['e_query']:
    exp_name+="_e_query"
if args['non_shadow']:
    exp_name+="_non_shadow"
if args['fixedBN']:
     exp_name+="_fixedBN"
if args['pre_con']:
     exp_name+="_precon"
exp_name+='_grid'+str(args['grid'])+'_beta'+str(args['beta'])

joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale'])),
    joint_transforms.RandomHorizontallyFlip()
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

train_set = ImageFolder(args,[ViSha_training_root], joint_transform, img_transform, target_transform,training=True)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=10, shuffle=True)

val_set = ImageFolder(args,[ViSha_validation_root], val_joint_transform, img_transform, target_transform)
val_loader = DataLoader(val_set, batch_size=args['train_batch_size'], num_workers=10, shuffle=False)

bce_logit = MyBceloss12_n().cuda()
bce_logit_dst = MyWcploss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = DSDNet(args)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])
   
    
    net = torch.nn.DataParallel(net).cuda().train()
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    best_mae = 100.0
    best_ber = 1000.0
    best_iter = curr_iter
    while True:
        # test_log, ber, mae = val(net, curr_iter)    
        train_loss_record_shad, loss_pc_record, loss_down1_record_shad = AvgMeter(), AvgMeter(), AvgMeter()
        loss_down2_record_shad, loss_down3_record_shad, loss_down4_record_shad = AvgMeter(), AvgMeter(), AvgMeter()

        train_loss_record_query, loss_fuse_record_dst1, loss_down1_record_dst1 = AvgMeter(), AvgMeter(), AvgMeter()
        loss_down2_record_dst1, loss_down3_record_dst1, loss_down4_record_dst1 = AvgMeter(), AvgMeter(), AvgMeter()

        train_loss_record_dst2, loss_fuse_record_dst2, loss_down1_record_dst2 = AvgMeter(), AvgMeter(), AvgMeter()
        loss_down2_record_dst2, loss_down3_record_dst2, loss_down4_record_dst2 = AvgMeter(), AvgMeter(), AvgMeter()
        train_loss_record = AvgMeter()
        net.train()
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            img, target, query_img, query_target = data['img'], data['target'], data['query_img'], data['query_target']
            inputs = dict()
            if args['e_query']:
                e_query_img =  data['e_query_img']
                e_query_img = Variable(e_query_img).cuda()
                inputs['e_query_img'] = e_query_img

            batch_size = img.size(0)
            img = Variable(img).cuda()
            target = Variable(target).cuda()

            query_img = Variable(query_img).cuda()
            query_target = Variable(query_target).cuda()
            
           
            inputs['img'] = img
            inputs['target'] = target
            inputs['query_img'] = query_img
            inputs['query_target'] = query_target
            
        
            optimizer.zero_grad()

        
            fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad,  \
            outputs \
             = net(inputs,curr_iter=curr_iter)

           
            bce_loss1 = binary_xloss(fuse_pred_shad, target)
            loss_hinge1 = lovasz_hinge(fuse_pred_shad, target)

            bce_loss2 = binary_xloss(pred_down1_shad, target)
            loss_hinge2 = lovasz_hinge(pred_down1_shad, target)

            bce_loss3 = binary_xloss(pred_down2_shad, target)
            loss_hinge3 = lovasz_hinge(pred_down2_shad, target)

            bce_loss4 = binary_xloss(pred_down3_shad, target)
            loss_hinge4 = lovasz_hinge(pred_down3_shad, target)

            bce_loss5 = binary_xloss(pred_down4_shad, target)
            loss_hinge5 = lovasz_hinge(pred_down4_shad, target)

            query_loss_shad = 0
            query_target_l = [query_target, query_target]
            if 'query' in outputs.keys():
                for i, query_l in  enumerate(outputs['query']):
                    for each_query_l in query_l:
                        # print(each_query_l.size(),query_target_l[i].size())
                        query_loss_shad = query_loss_shad + binary_xloss(each_query_l, query_target_l[i]) + lovasz_hinge(each_query_l, query_target_l[i])

            

            loss_shad = bce_loss1 + loss_hinge1 + bce_loss2 + loss_hinge2 + bce_loss3 +loss_hinge3 + bce_loss4 + loss_hinge4 + bce_loss5 + loss_hinge5 
             
            # loss = loss_shad 
           
            # loss_shad = loss_fuse_shad + loss1_shad + loss2_shad + loss3_shad + loss4_shad +loss0_shad
            con_loss = 0
            for sim_matrix in outputs['sim_matrix']:
                con_loss = con_loss + pc_loss(sim_matrix[0], sim_matrix[1]) + margin_loss(sim_matrix[0], sim_matrix[2],args['beta'])
            # con_loss = con_loss.sum()
            loss = loss_shad + query_loss_shad  + args['weight']*con_loss
            loss.backward()

            optimizer.step()
           
            train_loss_record.update(loss.data, batch_size)
            train_loss_record_shad.update(loss_shad.data, batch_size)

            train_loss_record_query.update(query_loss_shad.data, batch_size)
            loss_pc_record.update(con_loss.data, batch_size)
      

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [loss_train_shad %.5f], [loss_train_query %.5f], [loss_train_pc %.5f], [lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, train_loss_record_shad.avg, train_loss_record_query.avg,
                   loss_pc_record.avg,optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')
        
            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                return
        log=''
        test_log, ber, mae = val(net, curr_iter)
        log += test_log
        if best_mae > mae:
            best_mae = mae
            best_iter = curr_iter
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'best_mae_%d.pth' % best_iter))
        if best_ber > ber:
            best_ber = ber
            best_iter = curr_iter
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'best_ber_%d.pth' % best_iter))
        best_log = '[iter %d], [best iter %d], [best_mae %.5f], [current_mae %.5f], [current_ber %.5f]' % \
                (curr_iter, best_iter, best_mae,mae,ber)
        log +='\n' +best_log
        print(log)
        open(log_path, 'a').write(log + '\n')
        

def val(net, curr_iter):
    mae_record = AvgMeter()
    ber_record = AvgMeter()
    temporal_record = AvgMeter()
    val_loss_record = AvgMeter()
    val_loss_record_shad = AvgMeter()
    val_loss_record_query = AvgMeter()
    loss_pc_record = AvgMeter()
    net.eval()
    with torch.no_grad():
        for  data in tqdm((val_loader)):
            # img, target, query_img, query_target = data['img'], data['target'], data['query_img'], data['query_target']
            img, target, query_img, query_target = data['img'], data['target'], data['query_img'], data['query_target']
            
            # if args['brightness']:
            #     e_query_img =  data['e_query_img']
            #     e_query_img = Variable(e_query_img).cuda()
            #     inputs['e_query_img'] = e_query_img

            batch_size = img.size(0)

            img = Variable(img).cuda()
            target = Variable(target).cuda()

            query_img = Variable(query_img).cuda()
            query_target = Variable(query_target).cuda()
            
            # e_query_target = Variable(e_query_target).cuda()
            # labels_dst1 = Variable(labels_dst1).cuda()
            # labels_dst2 = Variable(labels_dst2).cuda()
            inputs = dict()
            inputs['img'] = img
            inputs['target'] = target
            inputs['query_img'] = query_img
            inputs['query_target'] = query_target
            
            # inputs['e_query_target'] = e_query_target
            # inputs['query_target'] = query_target
          
            # fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad,  \
            # query_fuse_pred, query_pred_down1, query_pred_down2, query_pred_down3, query_pred_down4,con_loss \
            #  = net(inputs)
          
         

            # inputs['e_query_img'] = e_query_img
            # inputs['e_query_target'] = e_query_target
            
            
            # print('sb')
            fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad,  \
            outputs\
             = net(inputs,val=True)
            # outputs = net(inputs)

            bce_loss1 = binary_xloss(fuse_pred_shad, target)
            loss_hinge1 = lovasz_hinge(fuse_pred_shad, target)

            bce_loss2 = binary_xloss(pred_down1_shad, target)
            loss_hinge2 = lovasz_hinge(pred_down1_shad, target)

            bce_loss3 = binary_xloss(pred_down2_shad, target)
            loss_hinge3 = lovasz_hinge(pred_down2_shad, target)

            bce_loss4 = binary_xloss(pred_down3_shad, target)
            loss_hinge4 = lovasz_hinge(pred_down3_shad, target)

            bce_loss5 = binary_xloss(pred_down4_shad, target)
            loss_hinge5 = lovasz_hinge(pred_down4_shad, target)

            query_loss_shad = 0
            query_target_l = [query_target, query_target]
            
            if 'query' in outputs.keys():
                for i, query_l in  enumerate(outputs['query']):
                    for each_query_l in query_l:
                        # print(each_query_l.size(),query_target_l[i].size())
                        query_loss_shad = query_loss_shad + binary_xloss(each_query_l, query_target_l[i]) + lovasz_hinge(each_query_l, query_target_l[i])
           
            

            loss_shad = bce_loss1 + loss_hinge1 + bce_loss2 + loss_hinge2 + bce_loss3 +loss_hinge3 + bce_loss4 + loss_hinge4 + bce_loss5 + loss_hinge5 
            
           
            con_loss = 0
            for sim_matrix in outputs['sim_matrix']:
                con_loss = con_loss + pc_loss(sim_matrix[0], sim_matrix[1]) 
            # con_loss = pc_c.sum()
            loss = loss_shad + query_loss_shad  + args['weight']*con_loss

            res = (fuse_pred_shad.data > 0).to(torch.float32)
            gt = (target.data > 0).to(torch.float32)
            fuse_next_pred = outputs['query'][0][0]
            res_1 = (fuse_next_pred.data > 0).to(torch.float32)
            # res = outputs
            prediction = np.array(
                       (res.cpu()))
            prediction_1 =    np.array(
                       (res_1.cpu()))    
            gt = np.array(gt.cpu().numpy())
            fuse = prediction 
            gt = gt 
            # print(prediction.shape,gt.shape)
            mae = np.mean(np.abs(fuse - gt))
            # print(mae)
            mae_record.update(mae)
            # print(prediction)
            # print(gt)
            # print(fuse.max(),gt.max())
            BER, shadow_BER, non_shadow_BER = cal_BER(prediction, gt)
            
            temporal_c = cal_temporal2(prediction, prediction_1)
            temporal_record.update(temporal_c,batch_size)

            # print(BER,shadow_BER,non_shadow_BER)
            ber_record.update(BER,batch_size)
            val_loss_record.update(loss.data, batch_size)
            val_loss_record_shad.update(loss_shad.data, batch_size)

            val_loss_record_query.update(query_loss_shad.data, batch_size)
            loss_pc_record.update(con_loss.data, batch_size)
    



          

        log = '[iter %d], [train loss %.5f], [loss_train_shad %.5f], [loss_train_query %.5f], [loss_train_pc %.5f], [mae %.5f], [ber %.5f], [tc %.5f]' % \
                  (curr_iter, val_loss_record.avg, val_loss_record_shad.avg, val_loss_record_query.avg,
                   loss_pc_record.avg, mae_record.avg, ber_record.avg,temporal_record.avg)
    return log, ber_record.avg, mae_record.avg

if __name__ == '__main__':
    main()
