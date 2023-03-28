from arm import ArmSE3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# ------------------------- Setup single 2link planar arm ------------------------- #
zero = np.eye(4)
zero[:3,3] = [2,0,0]

l = 1
w = np.array([
    [0,0,1],
    [0,0,1]
])
p = np.array([
    [0,0,0],
    [l,0,0]
])

screws = np.zeros((2,6))
for i in range(2):
    screws[i,3:] = -np.cross(w[i], p[i])
    screws[i,:3] = w[i]

arm = ArmSE3(screws, zero)


# ------------------------- Propagate particles through distribution ------------------------- #
N = 10000
var = [0.5, 0.5]
theta = np.array([
    0,
    0
])
thetas_noisy = np.random.normal(theta, np.sqrt(var), size=(N,2))

fk_true = arm.fk(theta)
fk_noisy = [arm.fk(t) for t in thetas_noisy]


# ------------------------- Sample from posterior distribution ------------------------- #
cov = np.zeros((6,6))
for i in range(2):
    adj_inv = np.linalg.inv( arm._adjoint( arm.fk(theta[i+1:], idx_start=i+1) ) )
    xi = adj_inv@arm.screws[i]
    cov += np.outer(xi, xi)*var[i]

noise = np.random.multivariate_normal(np.zeros(6), cov, size=N)
# print(fk_true)
fk_noisy_prop = [fk_true@arm._exp(w) for w in noise]


# ------------------------- Plot resulting trajectory ------------------------- #
fig, ax = plt.subplots(1, 2)

ax[0].set_title("Actual Distribution")
x = np.array([i[0,3] for i in fk_noisy] )
y = np.array([i[1,3] for i in fk_noisy] )
ax[0].scatter(x, y, s=0.5, alpha=0.5)

ax[1].set_title("Sampled Distribution")
x = np.array([i[0,3] for i in fk_noisy_prop] )
y = np.array([i[1,3] for i in fk_noisy_prop] )
ax[1].scatter(x, y, s=0.5, alpha=0.5)

ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

plt.setp(ax, xlim=ax[0].get_xlim(), ylim=ax[0].get_ylim())
plt.tight_layout()
plt.show()