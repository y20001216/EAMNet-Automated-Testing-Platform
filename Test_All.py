import torch
import cv2
import torch.nn.functional as F
import numpy as np
import os
import argparse
from lib.EAMNet import Network
from utils.sdy_data_val import test_dataset
from utils.evaluator import Evaluator
import shutil

print("--- 正在运行最终修正版 v3 (无归一化) ---")

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/EAMNet_E_b4_3/Net_epoch_best.pth')
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 数据集配置 ==================
dataset_config = {
    'CAMO': {
        'image_root': '/root/autodl-tmp/codzip/CAMO_prepared/test/image',
        'gt_root':    '/root/autodl-tmp/codzip/CAMO_prepared/test/mask'
    },
    'COD10K': {
        'image_root': '/root/autodl-tmp/codzip/COD10K/COD10K-v3/Test/Image',
        'gt_root':    '/root/autodl-tmp/codzip/COD10K/COD10K-v3/Test/GT_Object'
    },
    'NC4K': {
        'image_root': '/root/autodl-tmp/codzip/NC4K/Imgs',
        'gt_root':    '/root/autodl-tmp/codzip/NC4K/GT'
    }
}

model = Network()
model.load_state_dict(torch.load(opt.pth_path, map_location=device))
model.to(device)
model.eval()
results = []

with torch.no_grad():
    for name, paths in dataset_config.items():
        print(f"\n=== Testing on {name} ===")
        image_root, gt_root = paths['image_root'], paths['gt_root']

        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        save_base = f'./res/{opt.pth_path.split("/")[-2]}/{name}'
        save_path = os.path.join(save_base, 'preds')
        edge_save_path = os.path.join(save_base, 'edge')

        # 清空旧结果
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        if os.path.exists(edge_save_path):
            shutil.rmtree(edge_save_path)

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(edge_save_path, exist_ok=True)

        for i in range(test_loader.size):
            image, gt, name_img, _ = test_loader.load_data()
            image = image.to(device)

            outputs = model(image)
            main_prediction = outputs[1]  
            learned_edge = outputs[8]
            gt_np = np.array(gt)
            target_size = gt_np.shape[:2]

            # 保存预测图像
            def save_result(pred, path):
                pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
                pred = pred.sigmoid().detach().cpu().numpy().squeeze()
                if i == 0 and name == 'CAMO':
                     print(f"DEBUG: pred 范围: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
                # pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                pred = (255 * pred).astype(np.uint8)
                cv2.imwrite(path, pred)

            base_name = os.path.basename(name_img)
            save_result(main_prediction, os.path.join(save_path, base_name))
            save_result(learned_edge, os.path.join(edge_save_path, base_name))

        print(f"==> {name} Prediction Finished.")

        evaluator = Evaluator(pred_root=save_path, gt_root=gt_root)
        mae, f, s, e = evaluator.run()
        print(f"[{name}] MAE: {mae:.4f}, F-measure: {f:.4f}, S-measure: {s:.4f}, E-measure: {e:.4f}")
        results.append((name, mae, f, s, e))

# ================== 输出总结 ==================
print("\n=== Summary ===")
print("| Dataset | MAE | F-measure | S-measure | E-measure |")
print("|---------|------|------------|------------|------------|")
for r in results:
    print(f"| {r[0]:<7} | {r[1]:.4f} | {r[2]:.4f}     | {r[3]:.4f}     | {r[4]:.4f}     |")
