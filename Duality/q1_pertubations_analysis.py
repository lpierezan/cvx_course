import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

P = np.array([  [1, -1/2], 
                [-1/2, 2]])

q = np.array([-1, 0])

x = cp.Variable(2)

obj = cp.quad_form(x,P) + q.T @ x
u = cp.Parameter(2)
A = np.array([  [1,2],
                [1,-4],
                [1,1]])

constraints = [
    A[0] @ x <= u[0],
    A[1] @ x <= u[1],
    A[2] @ x >= -5
]

prob = cp.Problem(cp.Minimize(obj), constraints=constraints)

# (a)
u.value = np.array([-2,-3])
prob.solve()
print('status:', prob.status)
print('prob value:', prob.value)
lS = np.array([c.dual_value for c in constraints])
xS = np.array(x.value)
print('primal optimal', xS)
print('dual optimal:', lS)

# Check KKT
print('Checking KKT')
# primal feasible
assert np.array([c.value() for c in constraints]).all()
# dual feasible
assert (lS >= 0).all()
# comp. slackness
f = [c.expr.value for c in constraints]
assert np.isclose(lS @ f , 0)
# grad_x L(xS,lS) = 0
L_grad = 2*(P @ xS) + q + A.T @ lS
print('L grad', L_grad)
assert np.isclose(L_grad, 0).all()
print('KKT ok')

print('x2:', round(xS[1], 2))
print('lambda3:', round(lS[2],2))

# (b)
print('Pertubation table')

delta = [0, -0.1 ,0.1]
p_org = prob.value
dV = []
for d1 in delta:
    for d2 in delta:
        u.value = np.array([-2 + d1, -3 + d2])
        p_exact = prob.solve()
        assert prob.status == cp.OPTIMAL
        p_pred = p_org - lS.T @ [d1,d2,0]
        assert p_pred <= p_exact
        print(f'd1 {d1} d2 {d2} p_pred {p_pred} p_exact {p_exact} '
                f'exact - pred:{p_exact - p_pred}'
            )
        dV.append((p_exact - p_pred))

print('diff argsort', np.argsort(dV))




