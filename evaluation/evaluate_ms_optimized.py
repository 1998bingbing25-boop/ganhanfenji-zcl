"""
evaluate_ms_optimized.py
完整性能评估脚本 - 计算所有指标
"""
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_dataloaders

def evaluate_model(model_path, csv_path, data_root):
    """评估模型的所有性能指标"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 重建模型
    encoder = RestormerEncoder(
        inp_channels=8, dim=48, num_blocks=[4, 6],
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias'
    )
    
    class MSClassifier(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(48, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 5)
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.pool(x).view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = MSClassifier(encoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载验证数据
    print("加载验证数据...")
    _, val_loader = build_dataloaders(
        csv_path=csv_path,
        data_root=data_root,
        batch_size=16,
        num_workers=0,
        balanced=True,
        modalities=['ms']
    )
    
    # 预测
    print("进行预测...")
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for _, _, ms, labels in val_loader:
            ms = ms.to(device)
            outputs = model(ms)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # 计算指标
    print("\n" + "="*80)
    print("📊 完整性能评估")
    print("="*80)
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    print(f"\n总体指标:")
    print(f"  准确率 (Accuracy):  {accuracy*100:.2f}%")
    print(f"  精确率 (Precision): {precision*100:.2f}%")
    print(f"  召回率 (Recall):    {recall*100:.2f}%")
    print(f"  F1分数 (F1-Score):  {f1*100:.2f}%")
    
    # 与师姐对比
    print(f"\n与师姐成绩对比:")
    print(f"  {'指标':<15} {'我的':<15} {'师姐':<15} {'差距':<15}")
    print(f"  {'-'*60}")
    print(f"  {'准确率':<15} {accuracy*100:>6.2f}%{'':<7} {'89.19%':<15} {(accuracy-0.8919)*100:>+6.2f}%")
    print(f"  {'精确率':<15} {precision*100:>6.2f}%{'':<7} {'90.32%':<15} {(precision-0.9032)*100:>+6.2f}%")
    print(f"  {'召回率':<15} {recall*100:>6.2f}%{'':<7} {'96.55%':<15} {(recall-0.9655)*100:>+6.2f}%")
    print(f"  {'F1分数':<15} {f1*100:>6.2f}%{'':<7} {'93.33%':<15} {(f1-0.9333)*100:>+6.2f}%")
    
    # 混淆矩阵
    print(f"\n混淆矩阵:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)
    
    # 分类报告
    print(f"\n详细分类报告:")
    print(classification_report(all_targets, all_preds,
                               target_names=[f'Level {i}' for i in range(5)]))
    
    print("="*80)

if __name__ == '__main__':
    evaluate_model(
        'models_ms_opt_v1/drought_best.pth',
        '2025label_classic5.csv',
        'dataset/'
    )
