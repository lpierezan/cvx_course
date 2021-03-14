import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import matplotlib

"""
x = cp.Variable(1)
obj = cp.quad_form(x, [[1]]) + 1
constraints = [
    # (x-2)*(x-4) <= 0,
    cp.quad_form(x, [[1]]) - 6*x + 8 <= 0,
]
prob = cp.Problem(cp.Minimize(obj), constraints=constraints)
prob.solve()
"""
def f0(x):
    return x**2 + 1
def f1(x):
    return (x-2)*(x-4)
def L(x,l):
    return f0(x) + l*f1(x) 
"""
x^2 + 1 + l*(x^2 -6x + 8)
(l+1)x^2 -6*x*l + 1 + 8*l
min_x = 6*l / 2*(l+1) = 3l/(l+1)
l_min = -9l^2/(l+1) + 8l + 1
"""
def Lmin(l):
    xMin = 3*l/(l+1)
    return xMin , L(xMin, l)

x = np.linspace(-2,5,100)
obj = f0(x)
fig, ax = plt.subplots(figsize = (15,10))
ax.plot(x,obj, label = 'objective')
for i, l in enumerate([0.1,0.5,1,2,4,8]):
    ax.plot(x, L(x,l), label = f'lambda {l}', c = f"C{i+1}")
    xMin,lMin = Lmin(l)
    ax.scatter(xMin, lMin, s=80, marker = '*', c = f"C{i+1}")

ax.vlines([2,4], 0, max(obj), linestyles='dashed', label = 'feasible')
ax.set_ylim(0,max(obj))
ax.legend()
fig.show()

print(Lmin(2))

_ = input()
