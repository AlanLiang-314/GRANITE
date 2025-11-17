import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import entropy
import seaborn as sns

def compare_distributions(a, b, names=['A', 'B'], plot=False):
    """
    比較兩個形狀為(64,)的tensor的分布差異
    
    參數:
    a, b: 形狀為(64,)的tensor
    names: 兩個分布的名稱
    """
    # 確保輸入是numpy數組，便於計算
    if torch.is_tensor(a):
        a = a.cpu().detach().numpy()
    if torch.is_tensor(b):
        b = b.cpu().detach().numpy()
    
    results = {}
    
    # 1. 變異係數（CV）比較
    cv_a = np.std(a) / np.mean(a) * 100 if np.mean(a) != 0 else np.nan
    cv_b = np.std(b) / np.mean(b) * 100 if np.mean(b) != 0 else np.nan
    
    results['CV'] = {names[0]: cv_a, names[1]: cv_b}
    
    # 2. 分布形狀指標：偏度和峰度
    skew_a = stats.skew(a)
    skew_b = stats.skew(b)
    kurt_a = stats.kurtosis(a)
    kurt_b = stats.kurtosis(b)
    
    results['skew'] = {names[0]: skew_a, names[1]: skew_b}
    results['kurt'] = {names[0]: kurt_a, names[1]: kurt_b}
    
    # 3. 熵指標計算
    # 使用直方圖來估計概率分布
    hist_a, _ = np.histogram(a, bins=10, density=True)
    hist_b, _ = np.histogram(b, bins=10, density=True)
    
    # 確保概率和為1且沒有0值（避免log(0)問題）
    hist_a = hist_a + 1e-10
    hist_b = hist_b + 1e-10
    hist_a = hist_a / np.sum(hist_a)
    hist_b = hist_b / np.sum(hist_b)
    
    entropy_a = entropy(hist_a)
    entropy_b = entropy(hist_b)
    kl_div = entropy(hist_a, hist_b)  # KL散度
    
    results['entropy'] = {names[0]: entropy_a, names[1]: entropy_b}
    results['kldiv'] = kl_div
    
    if plot:
    
        # 4. 創建圖表
        fig, axs = plt.subplots(2, 3, figsize=(21, 10))
        
        # 4.1 直方圖比較
        axs[0, 0].hist(a, bins=15, alpha=0.5, label=names[0])
        axs[0, 0].hist(b, bins=15, alpha=0.5, label=names[1])
        axs[0, 0].set_title('hist')
        axs[0, 0].legend()
        
        # 4.2 核密度估計
        sns.kdeplot(a, ax=axs[0, 1], label=names[0])
        sns.kdeplot(b, ax=axs[0, 1], label=names[1])
        axs[0, 1].set_title('kde')
        axs[0, 1].legend()
        
        # 4.3 箱型圖
        box_data = [a, b]
        axs[1, 0].boxplot(box_data, labels=names)
        axs[1, 0].set_title('box')
        
        # 4.4 Q-Q圖
        stats.probplot(a, dist="norm", plot=axs[1, 1])
        axs[1, 1].set_title(f'Q-Q plot ({names[0]} vs norm dist.)')
        
        stats.probplot(b, dist="norm", plot=axs[0, 2])
        axs[0, 2].set_title(f'Q-Q plot ({names[0]} vs norm dist.)')

        plt.scatter(a, b)
        plt.plot([min(a.min(), b.min()), max(a.max(), b.max())], 
                [min(a.min(), b.min()), max(a.max(), b.max())], 'r--')
        axs[1, 2].set_title(f'Q-Q plot ({names[0]} {names[1]})')
        plt.grid(True)

        
        plt.tight_layout()
        plt.savefig('temp.png')
    
    # 返回結果和圖表
    return results

# 生成模擬數據進行演示
# 假設是兩組不同分布的數據

if __name__ == '__main__':
    np.random.seed(42)
    # 組A：均勻分布數據
    a = torch.tensor(np.random.normal(100, 10, 64))
    # 組B：一個極端值，其他集中的數據
    b = torch.tensor(np.random.normal(100, 5, 64))
    b[0] = 200  # 添加一個極端值

    # 執行比較分析
    results, fig = compare_distributions(a, b)

    # 印出結果
    for metric, values in results.items():
        if isinstance(values, dict):
            print(f"{metric}:")
            for name, value in values.items():
                print(f"  {name}: {value:.4f}")
        else:
            print(f"{metric}: {values:.4f}")

    # 顯示圖表
    plt.show()

    # 為了更清楚地比較兩個分布，再做一個Q-Q圖直接比較兩組數據
    plt.figure(figsize=(8, 8))
    stats.probplot(a.numpy(), dist="norm", plot=plt)
    plt.title('A分布的Q-Q圖')
    plt.figure(figsize=(8, 8))
    stats.probplot(b.numpy(), dist="norm", plot=plt)
    plt.title('B分布的Q-Q圖')

    # 直接比較兩組數據的Q-Q圖
    plt.figure(figsize=(8, 8))
    plt.scatter(np.sort(a.numpy()), np.sort(b.numpy()))
    plt.plot([min(a.min(), b.min()), max(a.max(), b.max())], 
            [min(a.min(), b.min()), max(a.max(), b.max())], 'r--')
    plt.xlabel('分布A分位數')
    plt.ylabel('分布B分位數')
    plt.title('A vs B的Q-Q圖')
    plt.grid(True)
    plt.show()