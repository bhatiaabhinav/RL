Optnet for DDPG Constrained Projection
=======================================

For ease of implementation, we will consider only recursive greedy projection. i.e. do the allocations top down in reference to the constraints tree.

Problem statement
-----------------
Given a given node $g$ (which has already been allocated $C$ resources) from the constraints tree, we want to allocate resources to it's $k$ children such that:

$$\sum_i^k z_i = C$$
$$\forall_i^k : 0 \le \check{C_i} \le z_i \le \hat{C_i} \le 1$$

We are given $C$ and a vector $\vec{y} \in [0, 1]^k$. We want to project this vector to the nearest feasible solution $\vec{z}$.

The linear program:
-------------------

$$\min \sum_i^k |z_i - y_i| \quad \text{subject to}$$
$$\sum_i^k z_i = C$$
$$\forall_i^k : z_i \le \hat{C_i}$$
$$\forall_i^k : \check{C_i} \le z_i$$

An equivalent linear program is:

$$
\begin{aligned}
\underset{\vec{d},\vec{z}}{\min} \sum_i^k d_i \quad &\text{subject to} \\
\sum_i^k z_i - C &= 0                   \qquad \lambda\\
\forall_i^k : y_i - z_i - d_i &\le 0      \qquad \mu_i\\
\forall_i^k : z_i - y_i - d_i &\le 0      \qquad \nu_i\\
\forall_i^k : z_i - \hat{C_i} &\le 0      \qquad \alpha_i\\
\forall_i^k : \check{C_i} - z_i &\le 0    \qquad \beta_i
\end{aligned}
$$ (1)

Here $\lambda,\mu_i,\nu_i,\alpha_i,\beta_i$ are the corresponding Lagrange multipliers.

The KKT Conditions
------------------

The Langragian is:

$$
\begin{aligned}
L(\vec{d},\vec{z},\vec{\mu},\vec{\nu},\vec{\alpha},\vec{\beta},\lambda) & = \sum_i^k d_i \\
& + \sum_i^k \mu_i(y_i - z_i - d_i) & + \sum_i^k \nu_i(z_i - y_i - d_i) \\
& + \sum_i^k \alpha_i(z_i - \hat{C_i}) & + \sum_i^k \beta_i(\check{C_i} - z_i) \\
& + \lambda (\sum_i^k z_i - C)
\end{aligned}
$$ (2)

The KKT conditions (conditions satisfied by the solution $\vec{d}^*, \vec{z}^*, \vec{\mu}^*, \vec{\nu}^*, \vec{\alpha}^*, \vec{\beta}^*, \lambda^*$ of the LP (1)) is given by

$$
\begin{aligned}
&\nabla_{\vec{d},\vec{z},\lambda} L & = \vec{0} \\
\forall_i^k: \quad &\mu_i(y_i - z_i - d_i) & = 0 \\
\forall_i^k : \quad &\nu_i(z_i - y_i - d_i) & = 0 \\
\forall_i^k : \quad &\alpha_i(z_i - \hat{C_i}) & = 0 \\
\forall_i^k : \quad &\beta_i(\check{C_i} - z_i) & = 0
\end{aligned}
$$

which expand to:

$$
\begin{aligned}
&\sum_i^k z_i - C & = 0 \\
\forall_i^k : \quad & 1 + \mu_i + \nu_i & = 0\\
\forall_i^k : \quad & -\mu_i + \nu_i + \alpha_i - \beta_i + \lambda & = 0 \\
\forall_i^k : \quad & \mu_i(y_i - z_i - d_i) & = 0 \\
\forall_i^k : \quad & \nu_i(z_i - y_i - d_i) & = 0 \\
\forall_i^k : \quad & \alpha_i(z_i - \hat{C_i}) & = 0 \\
\forall_i^k : \quad & \beta_i(\check{C_i} - z_i) & = 0
\end{aligned}
$$ (3)


Differentiating the KKT conditions
----------------------------------
We can differentiate both sides of each equation in set of equations (3) w.r.t to inputs $\vec{y}$ and $C$.  
The partial differential equations w.r.t. input $y_j$ are:

$$
\begin{aligned}
& \sum_i^k \frac{\partial z_i}{\partial y_j} & = 0 &\quad (a)\\

\forall_i^k : \quad & \frac{\partial\mu_i}{\partial y_j} + \frac{\partial \nu_i}{\partial y_j} & = 0 &\quad (b)\\

\forall_i^k : \quad & -\frac{\partial \mu_i}{\partial y_j} + \frac{\partial \nu_i}{\partial y_j} + \frac{\partial \alpha_i}{\partial y_j} - \frac{\partial \beta_i}{\partial y_j} + \frac{\partial \lambda}{\partial y_j} & = 0 &\quad (c)\\

\forall_i^k : \quad & \frac{\partial \mu_i}{\partial y_j}(y_i - z_i - d_i) + \mu_i (\delta_{ij} - \frac{\partial z_i}{\partial y_j} - \frac{\partial d_i}{\partial y_j}) & = 0 &\quad (d)\\

\forall_i^k : \quad & \frac{\partial \nu_i}{\partial y_j}(- y_i + z_i - d_i) + \nu_i(-\delta_{ij} + \frac{\partial z_i}{\partial y_j} - \frac{\partial d_i}{\partial y_j}) & = 0 &\quad (e)\\

\forall_i^k : \quad & \frac{\partial \alpha_i}{\partial y_j}(z_i - \hat{C_i}) + \alpha_i \frac{\partial z_i}{\partial y_j} & = 0 &\quad (f)\\

\forall_i^k : \quad & \frac{\partial \beta_i}{\partial y_j}(-z_i + \check{C_i}) - \beta_i \frac{\partial z_i}{\partial y_j}& = 0 &\quad (g)
\end{aligned}
$$ (4)

Here $\delta_{ij}$ is the Kronecker delta function, which is $1$ when $i=j$, and $0$ otherwise.

Partial differential equations w.r.t $C$ are:

$$
\begin{aligned}
& \sum_i^k \frac{\partial z_i}{\partial C} - 1 & = 0 \\

\forall_i^k : \quad & \frac{\partial\mu_i}{\partial C} + \frac{\partial \nu_i}{\partial C} & = 0\\

\forall_i^k : \quad & -\frac{\partial \mu_i}{\partial C} + \frac{\partial \nu_i}{\partial C} + \frac{\partial \alpha_i}{\partial C} - \frac{\partial \beta_i}{\partial C} + \frac{\partial \lambda}{\partial C} & = 0 \\

\forall_i^k : \quad & \frac{\partial \mu_i}{\partial C}(y_i - z_i - d_i) + \mu_i (- \frac{\partial z_i}{\partial C} - \frac{\partial d_i}{\partial C}) & = 0 \\

\forall_i^k : \quad & \frac{\partial \nu_i}{\partial C}(- y_i + z_i - d_i) + \nu_i(\frac{\partial z_i}{\partial C} - \frac{\partial d_i}{\partial C}) & = 0 \\

\forall_i^k : \quad & \frac{\partial \alpha_i}{\partial C}(z_i - \hat{C_i}) + \alpha_i \frac{\partial z_i}{\partial C} & = 0 \\

\forall_i^k : \quad & \frac{\partial \beta_i}{\partial C}(-z_i + \check{C_i}) - \beta_i \frac{\partial z_i}{\partial C}& = 0
\end{aligned}
$$ (5)

The equations can be solved independently per input $y_j$ and $C$.

Solving the system of equations (4) and (5)
-------------------------------------------

For equations (4), the variables are $\frac{\partial}{\partial y_j}$ of $\mu_i,\nu_i,\alpha_i,\beta_i,\lambda,d_i,z_i$. So there are $n=6k + 1$ variables and that many equations.

Trying to write equations (4) in matrix form:

$$A_{n \times n}J_{n \times 1}^{y_j} = B_{n \times 1}$$

Here $J^{y_j}$ is the column vector of derivates w.r.t $y_j$.

$$J^{y_j} = \frac{\partial}{\partial y_j}[\lambda,\mu_1,\nu_1,\alpha_1,\beta_1,d_1,z_1,\mu_2,...,z_2,...,u_k,...,z_k]^T$$

$$
A =
\begin{bmatrix}
eqn&\lambda & \mu_1 & \nu_1 & \alpha_1&\beta_1&d_1&z_1&\mu_2&...&z_2&...&\mu_k& \nu_k&\alpha_k&\beta_k&d_k&z_k\\
  a & 0 & 0 &  0 & 0 & 0 &  0 & 1 & 0 & ... & 1 & ... & 0 & 0 & 0 & 0 & 0 & 1 \\
b_1 & 0 &  1 & 1 & 0 &  0 & 0 & 0 & 0 & ... & 0 & ... & 0 & 0 & 0 & 0 & 0 & 0 \\
c_1 & 1 & -1 & 1 & 1 & -1 & 0 & 0 & 0 & ... & 0 & ... & 0 & 0 & 0 & 0 & 0 & 0 \\
d_1 & 0 & y_1-z_1-d_1 & 0 & 0 & 0 & -\mu_1 & -\mu_1 & 0 & ... & 0 & ... & 0 & 0 & 0 & 0 & 0 & 0 \\
e_1 & 0 & 0 & -y_1+z_1-d_1 & 0 & 0 & -\nu_1 & \nu_1 & 0 & ... & 0 & ... & 0 & 0 & 0 & 0 & 0 & 0 \\
f_1 & 0 & 0 & 0 & z_1-\hat{C_1} & 0 & 0 & \alpha_1 & 0 & ... & 0 & ... & 0 & 0 & 0 & 0 & 0 & 0 \\
g_1 & 0 & 0 & 0 & 0 & -z_1+\check{C_1} & 0 & -\beta_1 & 0 & ... & 0 & ... & 0 & 0 & 0 & 0 & 0 & 0 \\
b_2 & \vdots &&&&&&&&&&&&&&&&\vdots \\
\vdots & \vdots &&&&&&&&&&&&&&&&\vdots \\
g_k & 0 
\end{bmatrix}
$$

*Note: ignore the first row (heading row) and the first column (heading column) of $A$, which are there only for the ease of understanding the matrix. They are not a part of the matrix.*
*The heading row is telling what the respective columns are coefficients of in the set of linear quations (4).*
*The heading column indicates the sub-equation number from the set of equations (4)*

Basically,

$$
A_{rc} = \begin{cases}
1 & r=1,c=6m+1 & m=1,2,...,k \\
1 & r=6m-4,c=r,r+1 & m=1,2,...,k \\
1 & r=6m-3,c=1,r,r+1 & m=1,2,...,k \\
-1 & r=6m-3,c=r-1,r+2 & m=1,2,...,k \\
y_m-z_m-d_m & r=6m-2,c=r-2 & m=1,2...,k \\
-\mu_m & r=6m-2,c=r+2,r+3 & m=1,2,...,k \\
-y_m+z_m-d_m & r=6m-1,c=r-2 & m=1,2,...,k \\
-\nu_m & r=6m-1,c=r+1 & m=1,2,...,k \\
\nu_m & r=6m-1,c=r+2 & m=1,2,...,k \\
z_m-\hat{C_m} & r=6m,c=r-2 & m=1,2,...,k \\
\alpha_m & r=6m,c=r+1 & m=1,2,...,k \\
-z_m+\check{C_m} & r=6m+1,c=r-2 & m=1,2,...,k \\
-\beta_m & r=6m+1,c=r & m=1,2,...,k \\
0 & \text{otherwise}
\end{cases}
$$

and

$$
B_{r} = \begin{cases}
-\mu_m \delta_{mj}  & r=6m-2 && m=1,2,...,k \\
\nu_m \delta_{mj}   & r=6m-1 && m=1,2,...,k \\
0 & \text{otherwise}
\end{cases}
$$

Thus 

$$J^{y_j} = A^{-1}B$$

Similary, we can find $J^C=\frac{\partial}{\partial C}[\lambda,\mu_1,\nu_1,\alpha_1,\beta_1,d_1,z_1,\mu_2,...,z_2,...,u_k,...,z_k]^T$ by solving set of equations (5).

Then the overall Jacobian would be:

$$J_{(6k+1)\times (k+1)} = \begin{bmatrix}
J^{y_1}&J^{y_2}&...&J^{y_k}&J^C
\end{bmatrix}
$$

The overall picture
-------------------

From $J$, we can extract the rows corresponding to $z_i$ and traspose it and thus write $\nabla_{\vec{y},C}{\vec{z}}$, which is a $(k+1) \times k$ matrix.

Thus we can get gradient of output $\vec{z}$ w.r.t network parameters $\theta \in \mathbb{R}^p$ using the chain rule as:

$$(\nabla_{\theta}\vec{z})_{p \times k} = (\nabla_{\theta}(\vec{y},c))_{p \times (k+1)} (\nabla_{(\vec{y},C)}\vec{z})_{(k+1) \times k}$$