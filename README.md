# Installation

1. Clone the git repo:

```shell
git clone https://github.com/SingleZombie/DL-Demos.git
```

2. Run the installation command:

```shell
python setup.py develop
pip install -r requirements.txt
```

It is recommended to create a directory named `work_dirs` and put temporary results into it.

# Description

Demos for deep learning.

# Project

## Andrew Ng Deep Learning Specialization

01. Logistic Regression
02. Shallow Nerual Network
03. Deep Nerual Network (MLP)
04. Parameter Initialization
05. Regularization
06. Advanced Optimizer (mini-batch, momentum, Adam)
07. Multiclass Classification with TensorFlow and PyTorch
08. NumPy Convolution 2D
09. Basic CNN
10. ResNet
11. NMS
12. ~~My YOLO model~~
13. Letter level language model with PyTorch
14. Sentiment analysis using Glove with PyTorch
15. Date translation attention model with PyTorch
16. Transformer cn-en translation with PyTorch

## Generative Model

1. VAE with PyTorch
2. DDPM with PyTorch
3. PixelCNN with PyTorch
4. VQVAE with PyTorch
5. DDIM with PyTorch

## Others





最佳逼近问题（Best Approximation Problem）是数值分析和优化理论中的一个重要问题，其目标是在给定的函数空间或向量空间中，找到一个函数或向量，使得它与目标函数或向量之间的误差最小化。最佳逼近问题在许多领域中都有广泛的应用，如信号处理、图像处理、机器学习和数值计算等。

### 1. 最佳逼近问题的定义

假设我们有一个函数空间 \( V \) 和一个目标函数 \( f \)，我们希望在 \( V \) 中找到一个函数 \( g \)，使得 \( g \) 与 \( f \) 之间的误差最小化。误差通常用某种范数（如 \( L^2 \) 范数、\( L^\infty \) 范数等）来度量。

### 2. 最佳逼近问题的数学表达

最佳逼近问题可以表示为以下优化问题：

\[ \min_{g \in V} \| f - g \| \]

其中：
- \( \| \cdot \| \) 是某种范数，用于度量误差。
- \( f \) 是目标函数。
- \( g \) 是逼近函数，属于函数空间 \( V \)。

### 3. 常见范数

在最佳逼近问题中，常用的范数包括：

1. **\( L^2 \) 范数**：

\[ \| f - g \|_{L^2} = \left( \int |f(x) - g(x)|^2 \, dx \right)^{1/2} \]

2. **\( L^\infty \) 范数**：

\[ \| f - g \|_{L^\infty} = \sup_{x} |f(x) - g(x)| \]

3. **欧几里得范数**（在向量空间中）：

\[ \| \mathbf{f} - \mathbf{g} \|_2 = \sqrt{\sum_{i} (f_i - g_i)^2} \]

### 4. 最佳逼近问题的解法

最佳逼近问题的解法取决于函数空间 \( V \) 的性质和所使用的范数。以下是一些常见的解法：

#### 4.1 线性空间中的最佳逼近

如果 \( V \) 是一个有限维线性空间，最佳逼近问题可以通过求解线性方程组来解决。假设 \( V \) 的基向量为 \( \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n \)，我们可以将逼近函数 \( g \) 表示为基向量的线性组合：

\[ g = a_1 \mathbf{v}_1 + a_2 \mathbf{v}_2 + \cdots + a_n \mathbf{v}_n \]

最小化误差 \( \| \mathbf{f} - g \|_2 \) 可以通过求解正规方程（Normal Equation）得到：

\[ V^T V \mathbf{a} = V^T \mathbf{f} \]

其中：
- \( V \) 是基向量组成的矩阵：

\[ V = \begin{bmatrix}
\mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n
\end{bmatrix} \]

- \( \mathbf{a} \) 是系数向量：

\[ \mathbf{a} = \begin{bmatrix}
a_1 \\
a_2 \\
\vdots \\
a_n
\end{bmatrix} \]

#### 4.2 函数空间中的最佳逼近

在函数空间中，最佳逼近问题通常通过最小二乘法或正交投影来解决。假设 \( V \) 是一个函数空间，其基函数为 \( \phi_1(x), \phi_2(x), \dots, \phi_n(x) \)，我们可以将逼近函数 \( g \) 表示为基函数的线性组合：

\[ g(x) = a_1 \phi_1(x) + a_2 \phi_2(x) + \cdots + a_n \phi_n(x) \]

最小化误差 \( \| f - g \|_{L^2} \) 可以通过求解正规方程（Normal Equation）得到：

\[ \Phi^T \Phi \mathbf{a} = \Phi^T \mathbf{f} \]

其中：
- \( \Phi \) 是基函数组成的矩阵：

\[ \Phi = \begin{bmatrix}
\phi_1(x_1) & \phi_2(x_1) & \cdots & \phi_n(x_1) \\
\phi_1(x_2) & \phi_2(x_2) & \cdots & \phi_n(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_1(x_m) & \phi_2(x_m) & \cdots & \phi_n(x_m)
\end{bmatrix} \]

- \( \mathbf{a} \) 是系数向量：

\[ \mathbf{a} = \begin{bmatrix}
a_1 \\
a_2 \\
\vdots \\
a_n
\end{bmatrix} \]

- \( \mathbf{f} \) 是目标函数的值向量：

\[ \mathbf{f} = \begin{bmatrix}
f(x_1) \\
f(x_2) \\
\vdots \\
f(x_m)
\end{bmatrix} \]

### 5. 示例

假设我们有一个目标函数 \( f(x) = x^2 \)，我们希望在多项式空间 \( V = \text{span}\{1, x, x^2\} \) 中找到一个二次多项式 \( g(x) \)，使得 \( g(x) \) 与 \( f(x) \) 之间的 \( L^2 \) 误差最小化。

#### 5.1 基函数

基函数为 \( \phi_1(x) = 1 \)，\( \phi_2(x) = x \)，\( \phi_3(x) = x^2 \)。

#### 5.2 正规方程

构造矩阵 \( \Phi \) 和向量 \( \mathbf{f} \)：

\[ \Phi = \begin{bmatrix}
1 & x_1 & x_1^2 \\
1 & x_2 & x_2^2 \\
\vdots & \vdots & \vdots \\
1 & x_m & x_m^2
\end{bmatrix} \]

\[ \mathbf{f} = \begin{bmatrix}
f(x_1) \\
f(x_2) \\
\vdots \\
f(x_m)
\end{bmatrix} \]

求解正规方程 \( \Phi^T \Phi \mathbf{a} = \Phi^T \mathbf{f} \)，得到系数向量 \( \mathbf{a} \)。

#### 5.3 逼近函数

逼近函数 \( g(x) \) 为：

\[ g(x) = a_1 + a_2 x + a_3 x^2 \]

### 6. 总结

最佳逼近问题是在给定的函数空间或向量空间中，找到一个函数或向量，使得它与目标函数或向量之间的误差最小化。最佳逼近问题可以通过求解正规方程或使用最小二乘法来解决。最佳逼近问题在数值分析、信号处理、图像处理和机器学习等领域中具有广泛的应用。
1. Style Transfer with PyTorch
2. PyTorch DDP Demo
