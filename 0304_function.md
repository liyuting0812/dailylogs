## 0304_Function.

##### 1.赋予MAE与VIF各0.5的权重，计算得分，选择最小值。

##### 2.写成函数

> 拆分成几个不同的函数：
>
> * 调成月度数据
>
> * 删除‘不相关’
> * 删除‘连续为零’
> * 调参函数
> * 选择参数函数
>
> 输出结果
>
> 1.特征数据（xlsx）
>
> 2.相关系数图、时间序列图（pdf）

##### 3.使用石油ppi进行测试





```python

#pandas 显示所有行
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

