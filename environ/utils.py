# import sympy as sp

# # height = 50
# # 定义符号变量
# px, dpx, py = sp.symbols('px dpx py')

# # 定义 h(x) 函数
# def h(px, dpx, py):
#     y1 = sp.sqrt(px**2 + py**2)
#     y2 = sp.atan2(py, px)
#     return sp.Matrix([y1, y2])

# # 计算雅可比矩阵
# h_matrix = h(px, dpx, py)
# jacobian_matrix = h_matrix.jacobian([px, dpx, py])

# # 打印雅可比矩阵
# sp.pprint(jacobian_matrix)
