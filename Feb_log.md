## 1. Supervised learning

### 1.1 Linear Models 

Linear Regression fits a linear model with coefficients to minimize the residual sum of squares between the observed targets in the datasets, and the targets predicted by the linear approximation.



##### 特征工程

1. Filter：过滤法
   * 方差选择法
   * 相关系数法
   * 卡方检验
   * 互信息法

2. Wrapper：包装法
   + 递归特征消除法

3. Embedded：基于惩罚项的特征选择法
   + L1、L2惩罚项结合

##### 基于特征表示的迁移学习

利用源领域Ds和目标领域Dt的信息，寻找合适的特征表示空间作为所要迁移知识的载体，来增强目标领域的学习效果。关键在于如何找到合适的特征表示空间。

有监督学习方法借用深度学习中的微调（fine-tuning）方法，先用Ds样本训练一个模型，然后利用Dt中的样本，微调此基础模型



##### cross_val_score 函数，参数scoring设置

![img](https://upload-images.jianshu.io/upload_images/11409741-2c76b7c5e866f2cc.png?imageMogr2/auto-orient/strip|imageView2/2/w/740/format/webp)





###### 提升模型表现的做法

1. 预测目标改进，将预测目标值由连续变量改为非连续变量；
2. 使用多个模型来预测增加稳健性；
3. 把任务拆开，如先预测方向，再预测仓位。



#### 切换环境

```
conda info -e    #查看环境
conda activate pipeline   #切换环境
pip install 包 #下载包
conda config --add channels 镜像   #添加镜像
conda config --show  #看配置

```

tick_spacing = 20

from matplotlib.ticker import tick_spacing



2022.02.28

1. 选择Gap_Cross_Validation 方法中的GapWalkForward进行交叉验证。能够解决时间序列中数据不独立的问题。

```
gcv = GapWalkForward(n_splits =5, gap_size = 1,test_size = 6)
```

2. 选择四种回归方法，OLS，Lasso，Ridge，ElasticNet。

3. 使用GridSearchCV选择各模型的参数alpha。

4. 使用RFE(递归特征消除)方法进行特征数量选择，得到4个模型下最优的特征。

5. 4组特征进行4个模型的训练，选择平均MAE最小的特征作为特征集。

   

