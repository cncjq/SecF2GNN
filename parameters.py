import math

# S = 1
delta = 1e-5
epsilon = 200
q=1/10
T = 500
# sigma = (2 * math.log(1.25/delta)) ** 1/2 / epsilon
sigma = ((14*q**2*T*math.log(1/delta))/epsilon**2 + 7*q**2*T / epsilon)**(1/2)

print(sigma)
# print(sigma*q/500)

# glover = 14 * q * T * math.log(1/delta) / epsilon ** 2 + 7 * q ** 2 * T / epsilon

# print("sigma of glover is: ", glover)