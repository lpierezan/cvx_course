import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

A_file = "A.txt"
c_file = "c.txt"

A = np.loadtxt(A_file)
m, n = A.shape
c = np.loadtxt(c_file)
b = A@np.ones((n,1))/2
b = b.ravel()

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T@x),
                 [A @ x <= b, x >= 0, x <= 1])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
L = prob.value
print("A solution x is")
print(x.value)

assert (x.value >= 0).all()
assert (x.value <= 1).all()
assert (A@x.value <= b).all()

maxV = []
obj = []
ts = np.linspace(0,1,100)
for t in ts:
    x_ = (x.value >= t).astype(int)
    maxV.append(max(A@x_ - b))
    obj.append(c.T@x_)

maxV = np.array(maxV)
obj = np.array(obj)

fig , ax = plt.subplots()
ax.plot(ts, maxV, label = 'violation')
ax.plot(ts, obj, label = 'obj')
ax.legend()
fig.show()

U = min(obj[maxV <= 0])
print(f"L = {L} U = {U}")
print(f"gap = {U - L}")

_ = input()