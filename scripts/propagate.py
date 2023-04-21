import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../vjo")
from vjo.fk import ArmFK  # noqa E402


def plot_arm(theta, l, ax):  # noqa
    x = [0]
    y = [0]
    print(theta)
    x.append(l * np.cos(theta[0]))
    y.append(l * np.sin(theta[0]))

    x.append(x[-1] + l * np.cos(theta[0] + theta[1]))
    y.append(y[-1] + l * np.sin(theta[0] + theta[1]))

    ax.plot(x, y, marker="o", c="r")


# ------------------------- Setup single 2link planar arm ------------------------- #
"""
Arm is defined with x-axis right, y-axis up, z-axis out of page
2 link arm with one link at origin, another distance l away
Both links rotate about positive z-axis
"""
l = 1  # noqa
zero = np.eye(4)
zero[:3, 3] = [2 * l, 0, 0]

w = np.array([[0, 0, 1], [0, 0, 1]])
p = np.array([[0, 0, 0], [l, 0, 0]])

screws = np.zeros((2, 6))
for i in range(2):
    screws[i, 3:] = -np.cross(w[i], p[i])
    screws[i, :3] = w[i]

arm = ArmFK(screws, zero)


# ------------ Propagate particles through distribution ------------ #
N = 10000
var = [0.01, 0.01]
theta = np.array([np.pi / 4, np.pi / 4])
thetas_noisy = np.random.normal(theta, np.sqrt(var), size=(N, 2))

fk_true = arm.fk(theta)
fk_noisy = [arm.fk(t) for t in thetas_noisy]


# ---------- Sample from posterior distribution ---------- #
# # CORRECT WAY - No approximations to combine exponential for noise elements
# cov = np.zeros((6,6))
# noise = []
# for i in range(2):
#     adj_inv = np.linalg.inv( arm._adjoint( arm.fk(theta[i+1:], idx_start=i+1) ) )
#     xi = adj_inv@arm.screws[i]
#     cov = np.outer(xi, xi)*var[i]
#     noise.append( np.random.multivariate_normal(np.zeros(6), cov, size=N) )

# fk_noisy_prop = [fk_true@arm._exp(w1)@arm._exp(w2) for w1, w2 in zip(*noise)]

# APPROXIMATION - Combine matrix exponential which drops some higher order terms
# This approximation will matter less w/ smaller noises
# cov = np.zeros((6,6))
# for i in range(2):
#     adj_inv = np.linalg.inv( arm._adjoint( arm.fk(theta[i+1:], idx_start=i+1) ) )
#     xi = adj_inv@arm.screws[i]
#     cov += np.outer(xi, xi)*var[i]

cov = arm.fk_prop_cov(theta, var)

noise = np.random.multivariate_normal(np.zeros(6), cov, size=N)
fk_noisy_prop = [fk_true @ arm._exp(w) for w in noise]


# ------------------------- Plot resulting trajectory ------------------------- #
fig, ax = plt.subplots(1, 2, figsize=(7, 3))

ax[0].set_title("Actual Distribution")
x = np.array([i[0, 3] for i in fk_noisy])
y = np.array([i[1, 3] for i in fk_noisy])
ax[0].scatter(x, y, s=0.5, alpha=0.5)
plot_arm(theta, l, ax[0])

ax[1].set_title("Approximated Distribution")
x = np.array([i[0, 3] for i in fk_noisy_prop])
y = np.array([i[1, 3] for i in fk_noisy_prop])
ax[1].scatter(x, y, s=0.5, alpha=0.5)
plot_arm(theta, l, ax[1])

ax[0].set_aspect("equal")
ax[1].set_aspect("equal")

plt.setp(ax, xlim=ax[0].get_xlim(), ylim=ax[0].get_ylim())
plt.tight_layout()
plt.savefig("arm_cov_prop.png")
plt.show()
