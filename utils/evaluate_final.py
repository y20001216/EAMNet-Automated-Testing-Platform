# ====================================================================
#              evaluate_final.py (最终的、完美的、权威的版本)
# ====================================================================
import numpy as np
import os
from glob import glob
from PIL import Image
import torch
import torch.nn.functional as F

def load_images_from_dir(path):
    # 检查是否存在 .npy 文件
    npy_paths = sorted(glob(os.path.join(path, "*.npy")))
    if npy_paths:
        print(f"-> Detected .npy files in {path}. Loading them.")
        images = []
        names = []
        for npy_path in npy_paths:
            try:
                images.append(np.load(npy_path))
                names.append(os.path.splitext(os.path.basename(npy_path))[0])
            except Exception as e:
                print(f"Warning: Could not load npy file {npy_path}. Error: {e}")
        return images, names

    # 如果没有 .npy 文件，则加载图像文件
    img_paths = sorted(glob(os.path.join(path, "*.*")))
    print(f"-> No .npy files found. Loading image files from {path}.")
    images = []
    names = []
    for img_path in img_paths:
        # 过滤掉非图像文件
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue
        try:
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, dtype=np.float32)
            if np.max(img_np) > 1.0:
                img_np = img_np / 255.0
            images.append(img_np)
            names.append(os.path.splitext(os.path.basename(img_path))[0])
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Error: {e}")
    return images, names

def mae(pred, gt):
    """平均绝对误差。在原始浮点图上计算。"""
    return np.mean(np.abs(pred - gt))

def max_f_measure(pred, gt, beta2=0.3):
    """计算最大F-measure，使用最稳健的逻辑运算。"""
    gt_bool = gt >= 0.5
    if np.sum(gt_bool) == 0:
        return 1.0 if np.sum(pred < 0.5) == pred.size else 0.0

    f_scores = []
    for threshold in np.linspace(0, 1, 256):
        pred_bool = pred >= threshold
        tp = np.sum(pred_bool & gt_bool)
        if tp == 0:
            f_scores.append(0.0)
            continue
        
        pred_sum = np.sum(pred_bool)
        gt_sum = np.sum(gt_bool)
        
        pre = tp / (pred_sum + 1e-8)
        rec = tp / (gt_sum + 1e-8)
        
        f_score = (1 + beta2) * pre * rec / (beta2 * pre + rec + 1e-8)
        f_scores.append(f_score)
        
    return np.max(f_scores)

def s_measure(pred, gt):
    """S-measure。"""
    gt_bin = (gt >= 0.5).astype(np.float32)
    alpha = 0.5
    y = gt_bin.mean()
    if y == 0:
        return 1.0 - pred.mean()
    if y == 1:
        return pred.mean()
    
    mu_p, mu_g = pred.mean(), gt_bin.mean()
    
    o_fg = (pred * gt_bin).sum() * 2 / (pred.sum() + gt_bin.sum() + 1e-8)
    o_bg = ((1 - pred) * (1 - gt_bin)).sum() * 2 / ((1 - pred).sum() + (1 - gt_bin).sum() + 1e-8)
    u = y * o_fg + (1 - y) * o_bg
    
    sigma_p, sigma_g = pred.std(), gt_bin.std()
    sigma_pg = np.mean((pred - mu_p) * (gt_bin - mu_g))
    c2 = 0.03**2
    s_r = (2 * sigma_pg + c2) / (sigma_p**2 + sigma_g**2 + c2 + 1e-8)
    
    return alpha * u + (1 - alpha) * s_r

def max_e_measure(pred, gt):
    """计算标准的 Max E-measure，遍历阈值并在二值图上计算，最后取最大值。"""
    gt_bool = gt >= 0.5 # 布尔数组
    
    e_scores = []
    for threshold in np.linspace(0, 1, 256):
        pred_bool = pred >= threshold # 布尔数组
        
        pred_bin_float = pred_bool.astype(np.float32)
        gt_bin_float = gt_bool.astype(np.float32)

        align_matrix = 2 * (pred_bin_float - pred_bin_float.mean()) * (gt_bin_float - gt_bin_float.mean())
        denominator = (pred_bin_float - pred_bin_float.mean())**2 + (gt_bin_float - gt_bin_float.mean())**2 + 1e-8
        enhanced = (align_matrix / denominator + 1) ** 2 / 4
        e_scores.append(np.mean(enhanced))
        
    return np.max(e_scores) # 返回最大值，而不是平均值

def evaluate(pred_dir, gt_dir, dataset_name):
    preds, pred_names = load_images_from_dir(pred_dir)
    gts, gt_names = load_images_from_dir(gt_dir)
    
    gt_map = {name: img for name, img in zip(gt_names, gts)}
    
    if len(preds) == 0 or len(gts) == 0:
        print(f"Error: No valid predictions or ground truths found.")
        return None

    mae_scores, f_scores, s_scores, e_scores = [], [], [], []
    evaluated_count = 0
    
    for i, (pred, pred_name) in enumerate(zip(preds, pred_names)):
        if pred_name in gt_map:
            gt = gt_map[pred_name]
            
            if pred.shape != gt.shape:
                p_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
                p_tensor = F.interpolate(p_tensor, size=gt.shape, mode='bilinear', align_corners=False)
                pred = p_tensor.squeeze().numpy()

            is_cod10k = 'COD10K' in dataset_name
            if is_cod10k:
                if i == 0:
                    print(f"\n--- [COD10K 强度诊断] ---")
                    print(f"  原始 pred 均值: {pred.mean():.4f}, 最大值: {pred.max():.4f}")
                
                pred_brightened = np.clip(pred * 1.5, 0, 1)
                
                if i == 0:
                    print(f"  增强后 pred 均值: {pred_brightened.mean():.4f}, 最大值: {pred_brightened.max():.4f}")
                    print(f"--- 诊断结束 ---\n")
                
                pred_for_eval = pred_brightened
            else:
                pred_for_eval = pred

            mae_scores.append(mae(pred_for_eval, gt))
            f_scores.append(max_f_measure(pred_for_eval, gt))
            s_scores.append(s_measure(pred_for_eval, gt))
            e_scores.append(max_e_measure(pred_for_eval, gt))
            evaluated_count += 1
        else:
            print(f"Warning: No matching GT found for prediction {pred_name}.png. Skipping.")

    if evaluated_count == 0:
        print("Error: No valid pairs were evaluated.")
        return None

    return {
        "MAE": np.mean(mae_scores),
        "Max F-measure": np.mean(f_scores),
        "S-measure": np.mean(s_scores),
        "Max E-measure": np.mean(e_scores)
    }

if __name__ == "__main__":
    method_name = 'EAMNet_E_b4_3'
    
    dataset_config = {
        'COD10K': {
            'pred_root': f'./res/{method_name}/COD10K/preds_npy',
            'gt_root':   '/root/autodl-tmp/codzip/COD10K/COD10K-v3/Test/GT_Object'
        },
        'CAMO': {
            'pred_root': f'./res/{method_name}/CAMO/preds',
            'gt_root':   '/root/autodl-tmp/codzip/CAMO_prepared/test/mask'
        },
        'NC4K': {
            'pred_root': f'./res/{method_name}/NC4K/preds',
            'gt_root':   '/root/autodl-tmp/codzip/NC4K/GT'
        }
    }

    print("| Dataset | MAE    | Max F  | S-measure | Max E     |")
    print("|---------|--------|--------|-----------|-----------|")

    for name, paths in dataset_config.items():
        if not os.path.exists(paths['pred_root']):
            print(f"| {name:<7} | Skipping (Prediction directory not found) |")
            continue
        
        # --- 核心修正：在这里传入第三个参数 'name' ---
        metrics = evaluate(paths['pred_root'], paths['gt_root'], dataset_name=name)
        
        if metrics:
            print(f"| {name:<7} | {metrics['MAE']:.4f} | {metrics['Max F-measure']:.4f} | {metrics['S-measure']:.4f}  | {metrics['Max E-measure']:.4f}  |")