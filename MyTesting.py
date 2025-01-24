import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import test_dataset

def calc_mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def calc_sm(pred, gt):
    alpha = 0.5
    y = gt
    if y.mean() == 0:
        x = pred
        ps = 1.0
    else:
        x = pred
        ps = 2.0 * np.sum(x * y) / (np.sum(x * x) + np.sum(y * y))
    return ps


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/SINet_V2/Net_epoch_best.pth')
opt = parser.parse_args()

# for _data_name in ['CAMO', 'COD10K', 'CHAMELEON']:
data_path = './dataset/test/'
save_path = './res/{}/'.format(opt.pth_path.split('/')[-2])
model = Network(imagenet_pretrained=False)
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
image_root = '{}/image/'.format(data_path)
gt_root = '{}/GT/'.format(data_path)
test_loader = test_dataset(image_root, gt_root, opt.testsize)

total_mae = 0
total_sm = 0
results_file = os.path.join(save_path, 'metrics.txt')
sample_num = 0

# for i in range(test_loader.size):
#     image, gt, name, _ = test_loader.load_data()
#     gt = np.asarray(gt, np.float32)
#     gt /= (gt.max() + 1e-8)
#     image = image.cuda()

#     res5, res4, res3, res2 = model(image)
#     res = res2
#     res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
#     res = res.sigmoid().data.cpu().numpy().squeeze()
#     res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#     print('> processing - {}'.format(name))
#     #misc.imsave(save_path+name, res)
#         # If `mics` not works in your environment, please comment it and then use CV2
#     cv2.imwrite(save_path+name,res*255)

for i in range(test_loader.size):
    image, gt, name, img_for_save = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()

    res5, res4, res3, res2 = model(image)
    res = res2
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    
    # 将预测结果转换为uint8格式
    res = (res * 255).astype(np.uint8)
    # 将GT转换为uint8格式
    gt = (gt * 255).astype(np.uint8)
    # 确保原图为uint8格式
    img_for_save = np.array(img_for_save)
    
    # 转换为3通道(如果不是)
    if len(res.shape) == 2:
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    if len(gt.shape) == 2:
        gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        
    # 水平拼接
    concat_img = np.hstack((img_for_save, gt, res))
    # 计算指标
    pred_norm = res / 255.0  # 归一化到[0,1]
    gt_norm = gt / 255.0
    mae = calc_mae(pred_norm, gt_norm)
    sm = calc_sm(pred_norm, gt_norm)
    
    total_mae += mae
    total_sm += sm
    sample_num += 1
    
    print(f'> {name} - MAE: {mae:.4f}, S-measure: {sm:.4f}')
    print('> processing - {}'.format(name))
    cv2.imwrite(os.path.join(save_path, name), concat_img)

# 计算平均指标
avg_mae = total_mae / sample_num
avg_sm = total_sm / sample_num

# 保存结果
with open(results_file, 'w') as f:
    f.write(f'Average MAE: {avg_mae:.4f}\n')
    f.write(f'Average S-measure: {avg_sm:.4f}\n')