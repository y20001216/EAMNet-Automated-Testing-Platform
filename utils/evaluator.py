import numpy as np
import cv2
import os

class Evaluator:
    def __init__(self, pred_root, gt_root):
        self.pred_root = pred_root
        self.gt_root = gt_root
        self.file_names = sorted([f for f in os.listdir(gt_root) if f.endswith(('.png', '.jpg'))])

    def _read_image(self, path, grayscale=True):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        return img.astype(np.float32) / 255.0 if img is not None else None

    def mae(self, pred, gt):
        return np.mean(np.abs(pred - gt))

    def f_measure(self, pred, gt, beta2=0.3):
        precision, recall = [], []
        for t in np.linspace(0, 1, 256):
            bin_pred = (pred >= t).astype(np.float32)
            tp = (bin_pred * gt).sum()
            pre = tp / (bin_pred.sum() + 1e-8)
            rec = tp / (gt.sum() + 1e-8)
            precision.append(pre)
            recall.append(rec)
        precision = np.array(precision)
        recall = np.array(recall)
        return ((1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)).max()

    def e_measure(self, pred, gt):
        e_list = []
        for t in np.linspace(0, 1, 256):
            bin_pred = (pred >= t).astype(np.float32)
            align_matrix = 2 * (bin_pred - bin_pred.mean()) * (gt - gt.mean())
            denominator = (bin_pred - bin_pred.mean())**2 + (gt - gt.mean())**2 + 1e-8
            enhanced = (align_matrix / denominator + 1) ** 2 / 4
            e_list.append(enhanced.mean())
        return np.max(e_list)

    # 简化版 S-measure，可换成更精确版本
    def s_measure(self, pred, gt):
        s_object = 1 - np.mean((pred - gt) ** 2)
        s_region = self._s_region(pred, gt)
        return 0.5 * s_object + 0.5 * s_region

    def _s_region(self, pred, gt):
        X, Y = gt.shape
        x, y = int(X / 2), int(Y / 2)
        gt1, gt2, gt3, gt4 = gt[:x, :y], gt[:x, y:], gt[x:, :y], gt[x:, y:]
        pred1, pred2, pred3, pred4 = pred[:x, :y], pred[:x, y:], pred[x:, :y], pred[x:, y:]

        def ssim(a, b):
            mu_a, mu_b = a.mean(), b.mean()
            sigma_a, sigma_b = a.var(), b.var()
            sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
            return (2 * mu_a * mu_b + 0.01) * (2 * sigma_ab + 0.03) / ((mu_a**2 + mu_b**2 + 0.01) * (sigma_a + sigma_b + 0.03))

        return 0.25 * (ssim(pred1, gt1) + ssim(pred2, gt2) + ssim(pred3, gt3) + ssim(pred4, gt4))
    def run(self):
        mae_all, f_all, s_all, e_all = [], [], [], []
        
        debug_done_cod10k = False
        is_cod10k = 'COD10K' in self.gt_root
        
        for name in self.file_names:
            pred_path = os.path.join(self.pred_root, name)
            gt_path = os.path.join(self.gt_root, name)
            
            # 1. 先加载两个图像
            pred = self._read_image(pred_path)
            gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            # 2. 检查加载是否成功
            if pred is None or gt_raw is None:
                print(f"[Warning] Skipping file pair due to loading error: {name}")
                continue

            # 3. 如果是 COD10K，执行终极诊断
            if is_cod10k and not debug_done_cod10k:
                print("\n--- COD10K GT 终极诊断 ---")
                print(f"GT 路径: {gt_path}")
                print(f"GT 数组形状: {gt_raw.shape}")
                print(f"GT 数据类型: {gt_raw.dtype}")
                
                unique_values = np.unique(gt_raw)
                print(f"GT 数组中的唯一值: {unique_values}")
                
                print("--- 诊断结束 ---\n")
                debug_done_cod10k = True
                
            # 4. 执行最稳健的二值化
            gt = (gt_raw > 128).astype(np.float32)
            
            # 5. 确保尺寸一致并计算指标
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
                
            mae_all.append(self.mae(pred, gt))
            f_all.append(self.f_measure(pred, gt))
            s_all.append(self.s_measure(pred, gt))
            e_all.append(self.e_measure(pred, gt))
            
        return np.mean(mae_all), np.mean(f_all), np.mean(s_all), np.mean(e_all)