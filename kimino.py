from sympy import symbols, solve, cosh, sinh, cos, sin

# 変数の定義
A, B, x, L, EI, m, omega = symbols('AB B x L EI m omega')
C, D, E, F = symbols('C D E F')

# 与式の定義
eq1 = AB**2 * cosh(B*x)*C + AB**2 * sinh(B*x)*D - AB**2*cos(B*x)*E - AB**2*sin(B*x)*F=0
eq2 = AB**3 * sinh(B*x)*C + AB**3 * cosh(B*x)*D + AB**3*sin(B*x)*E - AB**3*cos(B*x)*F=0
eq3 = AB**2 * cosh(B*L)*C + AB**2 * sinh(B*L)*D - AB**2*cos(B*L)*E - AB**2*sin(B*L)*F=0
eq4 = EI*(AB**3 * sinh(B*L)*C + AB**3 * cosh(B*L)*D + AB**3*sin(B*L)*E - AB**3*cos(B*L)*F) + m*omega**2 * cosh(B*L)*C + m*omega**2 * sinh(B*L)*D + m*omega**2 * cos(B*L)*E + m*omega**2 * sin(B*L)*F=0

# 連立方程式を解く
solution = solve((eq1, eq2, eq3, eq4), (C, D, E, F))
print(solution)