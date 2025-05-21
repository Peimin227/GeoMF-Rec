

# GeoMF-Rec Documentation

## 1. 方法原理

GeoMF 的核心思路是在经典矩阵分解的协同过滤之上，额外引入地理偏好分量，将两者相加得到最终预测：
\[
\hat r_{u,i}
= \underbrace{P_u^\top Q_i}_{\text{协同过滤 CF 分量}}
\;+\;
\underbrace{X_u^\top Y_i}_{\text{地理偏好 Geo 分量}}
\]

1. **CF 分量**  
   - 对应公式 \(\hat r^{CF}_{u,i}=P_u^\top Q_i\)。  
   - \(P\in\mathbb R^{M\times K}\)，\(Q\in\mathbb R^{N\times K}\) 分别是用户和商户的 \(K\) 维潜在因子。  
   - 利用加权交替最小二乘（ALS）一步步对每个用户 \(u\) 和商户 \(i\) 解闭式最小二乘，固定 \(X, Y\) 后更新 \(P, Q\)。

2. **Geo 分量**  
   - 对应公式 \(\hat r^{Geo}_{u,i}=X_u^\top Y_i\)。  
   - \(Y\in\mathbb R^{N\times L}\) 是商户到 \(L\) 个地理格子的影响矩阵，由商户经纬度与格子中心距离经过高斯核计算而来。  
   - \(X\in\mathbb R^{M\times L}\) 是用户对这些格子的偏好分布，使用投影梯度（Projected Gradient）+ L1 正则逐行更新，并截断为非负。

3. **混合优化**  
   - **第 1 阶段**：严格版 GeoMF（ALS 更新 \(P, Q\) + 投影梯度更新 \(X\)），最小化加权 MSE + 正则。  
   - **第 2 阶段**：BPR 微调，针对 Top-K 排序直接优化负采样下的对比损失，用多进程 DataLoader 批量化加速。

最终预测：
\[
\hat r_{u,i}
= P_u^\top Q_i + X_u^\top Y_i
\]

---

## 2. 整体逻辑图

```mermaid
flowchart LR
  subgraph 数据预处理
    A1[review.json + tip.json] -->|inter_matrix.py| R[R 矩阵]
    A1 -->|inter_matrix.py| W[W 矩阵]
    A2[business.json] -->|grid.py| G[格子 Center & biz2grid]
    G -->|influence.py| Y[Y 矩阵]
  end

  subgraph 划分数据
    R & W -->|split.py| R_train[训练 R_train]
    R & W -->|split.py| R_test[测试 R_test]
  end

  subgraph 第1阶段：严格 GeoMF
    R_train & W_train & Y -->|train.py:<br>GeoMFPTStrict.fit()<br>(ALS + PG with tqdm)| P,Q,X
  end

  subgraph 第2阶段：BPR 微调
    R_train & P,Q,X -->|train.py:bpr_fine_tune()<br>DataLoader + Adam| P',Q',X'
  end

  subgraph 评估
    P',Q',X' & R_test -->|evaluate.py| 评估指标[Recall@K / Precision@K]
  end
```

---

## 3. 用到的 Feature 及在模型中的作用

| Feature            | 来源脚本／文件         | 矩阵符号                  | 处理方式／含义                                           | 在模型中的角色                        |
|--------------------|------------------------|---------------------------|--------------------------------------------------------|---------------------------------------|
| Review 交互        | `inter_matrix.py`      | \(R\)                     | 用户–商户是否有评论/Tip，二值化构稀疏矩阵               | 预测目标 \(\hat r\) 的基准             |
| 交互权重           | `inter_matrix.py`      | \(W\)                     | \(1+\alpha\ln(1+\text{count}_{ui})\)                    | ALS 中加权最小二乘的权重               |
| 商户经纬度         | `business.json`        | —                         | 解析经纬度，映射到格子中心后算距离                       | 生成地理影响矩阵 \(Y\)                 |
| 地理格子网格       | `grid.py`              | —                         | 划分经纬度平面为 \(L\) 个网格，输出网格中心坐标           | 空间编码基准，用于高斯核                |
| 地理影响矩阵       | `influence.py`         | \(Y\)                     | \(Y_{i,c}=\exp(-d(i,c)^2/2\sigma^2)\)                   | Geo 分量预测：\(X_u^\top Y_i\)          |
| 训练集交互         | `split.py`             | \(R_{\rm train}\)         | 留出法 per-user split，至少保留 1 条测试交互             | 严格 GeoMF & BPR 训练输入             |
| 测试集交互         | `split.py`             | \(R_{\rm test}\)          | 用户–商户测试交互                                       | Recall/Precision 评估                 |
| 用户潜因子         | `geo_mf.py` (fit)      | \(P\in\mathbb R^{M\times K}\) | ALS 闭式解得到                                        | CF 分量                               |
| 商户潜因子         | `geo_mf.py` (fit)      | \(Q\in\mathbb R^{N\times K}\) | ALS 闭式解得到                                        | CF 分量                               |
| 用户地理偏好       | `geo_mf.py` (fit)      | \(X\in\mathbb R^{M\times L}\) | 投影梯度 + L1 正则                                     | Geo 分量                              |
| BPR 排序微调       | `train.py` (bpr_fine_tune) | —                     | 多负采样 + Adam + DataLoader 批量化                     | Top-K 推荐效果提升                     |