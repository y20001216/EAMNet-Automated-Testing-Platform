
import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.EAMNet import Network
from utils.sdy_data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
from utils.loss_function import *
import logging
import torch.backends.cudnn as cudnnb
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def get_sobel_gradients(image):
    # image 应该是 [B, 1, H, W] 的张量
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(image.device)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(image.device)
    
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)


# def structure_loss(pred, mask):
#     """
#     loss function (ref: F3Net-AAAI-2020)
#     """
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask) * weit).sum(dim=(2, 3))
#     union = ((pred + mask) * weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1) / (union - inter + 1)
#     return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, writer,cur_loss):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, grads) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            grads = grads.cuda()
            #edge = edge.cuda()

            preds_tuple = model(images)
            preds = list(preds_tuple) # Convert to list to pop
            final_gate = preds.pop()          
            learned_edge_pred = preds.pop()

            pred_final = preds[1]
            pred_final_sigmoid = torch.sigmoid(pred_final)
            pred_grads = get_sobel_gradients(pred_final_sigmoid)
            gt_grads = get_sobel_gradients(gts)

            # 定义超参数
            warmup_boundary_epoch = 15 # 在前15个epoch，完全不使用boundary loss
            ramp_up_epochs = 15      # 在接下来的15个epoch，权重从0线性增长到1.0
            if epoch <= warmup_boundary_epoch:
                boundary_weight = 0.0
            else:
                # 计算 ramp-up 过程中的权重
                progress = (epoch - warmup_boundary_epoch) / ramp_up_epochs
                boundary_weight = min(1.0, progress) * 1.0 # 乘以最终的目标权重 1.0

            loss_boundary = F.l1_loss(pred_grads, gt_grads)

            gt_grads_mean = torch.mean(torch.abs(gt_grads)) + 1e-8
            loss_boundary = loss_boundary / gt_grads_mean

            loss_edge_supervision = F.binary_cross_entropy_with_logits(learned_edge_pred, grads)
            
            loss_grad = 0.2 * dice_loss(preds[3], grads) + 0.3 * dice_loss(preds[4], grads) + 0.5 * dice_loss(preds[5],grads)
            loss_init = 0.25 * cur_loss(preds[0], gts) + 0.5 * cur_loss(preds[2], gts)
            loss_final = 1.0 * cur_loss(preds[1], gts)
            gate_entropy_loss = -torch.mean(final_gate * torch.log(final_gate + 1e-8))
            
            loss = loss_init + loss_final + loss_grad + 0.2 * loss_edge_supervision \
                   + 0.05 * gate_entropy_loss + boundary_weight * loss_boundary

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            ema.update()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} loss_grad: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data, loss_grad.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} '
                    'Loss2: {:0.4f} Loss3: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data, loss_grad.data))
                # TensorboardX-Loss
                # writer.add_scalars('Loss_Statistics',
                #                    {'Loss_init': loss_init.data, 'Loss_final': loss_final.data, 'Lose_grad': loss_grad.data,
                #                     'Loss_total': loss.data},
                #                    global_step=step)
                # 正确的写法：将每一项损失作为独立的标量，记录在 'Loss' 分组下
                writer.add_scalar('Loss/Total_Step', loss.data, global_step=step)
                writer.add_scalar('Loss/Init_Step', loss_init.data, global_step=step)
                writer.add_scalar('Loss/Final_Step', loss_final.data, global_step=step)
                writer.add_scalar('Loss/Grad_Step', loss_grad.data, global_step=step)
                 # TensorboardX-Training Data
                # grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('RGB', grid_image, step)
                # grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('GT', grid_image, step)

                # # TensorboardX-Outputs
                # res = preds[0][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_init', torch.tensor(res), step, dataformats='HW')
                # res = preds[1][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')
       

        loss_all /= epoch_step
        #logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        #writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        loss_avg = loss_all / epoch_step
        print(f"Epoch [{epoch:03d}/{opt.epoch:03d}], Train Loss AVG: {loss_avg:.4f}")
        writer.add_scalar('Loss/train_avg', loss_avg, epoch)

        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """ 
    validation function
    """
    global best_mae, best_epoch
    
    ema.store()
    ema.copy_to() 
    
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            preds_all = model(image)
            pred_f = preds_all[1]
            fuse_3 = preds_all[6]
            fuse_2 = preds_all[7]
            learned_edge = preds_all[8]

            res = F.interpolate(pred_f, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            if i == 0:
                fm3 = fuse_3[0].detach().cpu()
                heatmap3 = torch.mean(fm3, dim=0).numpy()
                
                fm2 = fuse_2[0].detach().cpu()
                heatmap2 = torch.mean(fm2, dim=0).numpy()

                edge_vis = learned_edge[0].sigmoid().detach().cpu().squeeze().numpy()
                
                fig, axs = plt.subplots(1, 5, figsize=(25, 5))

                axs[0].imshow(img_for_post); axs[0].set_title('Original Image'); axs[0].axis('off')
                axs[1].imshow(gt, cmap='gray'); axs[1].set_title('Ground Truth'); axs[1].axis('off')
                axs[2].imshow(edge_vis, cmap='gray'); axs[2].set_title('Learned Edge'); axs[2].axis('off')
                axs[3].imshow(res, cmap='gray'); axs[3].set_title('Prediction'); axs[3].axis('off')
                axs[4].imshow(heatmap3, cmap='viridis'); axs[4].set_title('Final Fusion (fuse_3)'); axs[4].axis('off')

                plt.suptitle(f'Validation - Epoch {epoch}')

                # 创建一个专门用于存放验证结果图像的目录
                val_image_path = os.path.join(save_path, 'validation_images')
                os.makedirs(val_image_path, exist_ok=True)

                figure_save_path = os.path.join(val_image_path, f'epoch_{epoch:03d}.png')
                plt.savefig(figure_save_path)
                print(f"Saved validation visualization to {figure_save_path}")
                plt.close() # 关闭图像，防止占用过多内存
        mae = mae_sum / test_loader.size
        writer.add_scalar('Metric/val_MAE', mae, epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                print(f"Saving best EMA model state_dict at epoch {epoch}...")
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
        ema.restore()
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str,
                        default='/root/autodl-tmp/codzip/train_all',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='/root/autodl-tmp/codzip/CAMO_prepared/test',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                    default='./snapshot/EAMNet_E_b4_3/', 
                    help='the path to save model and log')
    parser.add_argument('--optimizer', type=str,
                    default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--loss_type', type=str, default='bei', help='the type of loss function')
    opt = parser.parse_args()

    # loss selection
    if opt.loss_type == 'bei':
        cur_loss = hybrid_e_loss
    elif opt.loss_type == 'wce':
        cur_loss = wce_loss
    elif opt.loss_type == 'wiou':
        cur_loss = wiou_loss
    elif opt.loss_type == 'e':
        cur_loss = e_loss
    else:
        raise Exception('No Type Matching')


    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    cudnn.benchmark = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # build the model
    model = Network().cuda()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.98)

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    grad_loss_func = torch.nn.MSELoss()
    #grad_loss_func = CharbonnierLoss

    params = model.parameters()

    print("Optimizer: AdamW with weight_decay=0.05")
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    import os

    train_loader = get_loader(
        image_root=os.path.join(opt.train_root, 'image'),
        gt_root=os.path.join(opt.train_root, 'mask'),
        grad_root=os.path.join(opt.train_root, 'edge'),
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=8)
    val_loader = test_dataset(image_root=os.path.join(opt.val_root, 'image'),
                              gt_root=os.path.join(opt.val_root, 'mask'),
                              grad_root=os.path.join(opt.val_root, 'edge'),
                              testsize=opt.trainsize)

    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    warmup_epochs = 8  
    print(f"LR Scheduler: CosineAnnealingLR with {warmup_epochs} warmup epochs.")
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch - warmup_epochs, eta_min=3e-6)

    print("Start train...")
    for epoch in range(1, opt.epoch + 1):
    # ---- 学习率 Warmup 逻辑 ----
        if epoch <= warmup_epochs:
            current_lr = opt.lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        elif epoch == warmup_epochs + 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr
    
    # ---- 正常的训练和验证流程 ----
        train(train_loader, model, optimizer, epoch, save_path, writer, cur_loss)
        val(val_loader, model, epoch, save_path, writer)

    # ---- 调度器更新逻辑 ----
    # Warmup 结束后，让 Cosine scheduler 接管学习率的调整
        if epoch > warmup_epochs:
            scheduler.step()
    
    # 可选：打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, Current LR: {current_lr}")
        writer.add_scalar('learning_rate', current_lr, global_step=epoch)
