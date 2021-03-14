import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


S = np.loadtxt("S.txt")
pMean = np.loadtxt("pbar.txt")
n = pMean.shape[0]

#%%
# What is the risk of the uniform portfolio?

xU = np.ones(n)/n
riskU = xU.T @ S @ xU

print(f"uniform portfolio risk: {riskU**0.5}")

#%%
# What is the risk of an optimal portfolio with no (additional) constraints?
print("min risk with expected return equals uniform portfolio")

x = cp.Variable(n)
constraints = \
[ 
    cp.sum(x) == 1, 
    pMean.T @ x == pMean.T @ xU 
]

prob = cp.Problem(cp.Minimize(cp.quad_form(x, S)), constraints)
prob.solve()
assert prob.status == cp.OPTIMAL
print(f"optimal risk {prob.value**0.5}")

#%%
# What is the risk of a long-only portfolio 𝑥⪰0?
print("x >= 0")
prob = cp.Problem(cp.Minimize(cp.quad_form(x, S)), constraints + [x >= 0])
prob.solve()
assert prob.status == cp.OPTIMAL
print(f"optimal risk {prob.value**0.5}")

#%%
# What is the risk of a portfolio with a limit on total short position:  1𝑇(𝑥−)≤0.5 , where  (𝑥−)𝑖=max{−𝑥𝑖,0} ?
print("limit on short")
prob = cp.Problem(cp.Minimize(cp.quad_form(x, S)), constraints + [cp.sum(cp.neg(x)) <= 0.5])
prob.solve()
assert prob.status == cp.OPTIMAL
print(f"optimal risk {prob.value**0.5}")
