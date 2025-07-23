# Natural Gradient Gaussian Approximation (NANO) Filter

This is the official python code for the paper "Nonlinear Bayesian Filtering with Natural Gradient Gaussian Approximation". You can find the preprint of the paper in [Link](https://arxiv.org/pdf/2410.15832v1). 

Please contact the corresponding author of the code at cwh19@mails.tsinghua.edu.cn or mehdizhang@126.com.

## 1. Installation

1. Clone the repository.

```bash
git clone https://github.com/TianyiMehdi/NANO-idlab.git
```

2. Install the dependencies.

```bash
pip install -r requirements.txt
```

## 2. Usage for new environ

1. Please prepare the state transition function, measurement function, and the Jacobian matrix of the measurement function, and place them in a new Python file within the environ/ folder. You can refer to the writing style of other models in this folder. The parameters needed for model include

```python
# In environ/your_model.py
self.dim_x = 3  # 状态维度
self.dim_y = 3  # 观测维度
self.P0 = np.eye(self.dim_x) * 5  # 初始状态协方差
self.m0 = np.array([5., 5., 5.])  # 初始状态均值
```

2. The filter/ directory does not need to be modified in any way and can be directly instantiated through the model.

```python
from filter import EKF, NANO, UKF
from environ import your_model
model = your_model()
filter = NANO(model, **args) # args代表其他需要的参数
```

3. The data required to run the filter includes the control input u and measurements y.

```python
for i in range(1, steps):
  filter.predict(u)
  filter.update(y)
  x_hat_list.append(filter.x) # 每个时间步获得的状态估计
```

