import math

def rdp_epsilon_approx(q, sigma, steps, delta):
    orders = [2, 3, 4, 5, 8, 16, 32, 64, 128, 256]
    log_inv_delta = math.log(1.0 / delta)

    best_eps = float("inf")
    best_order = None

    for alpha in orders:
        rdp = (q * q * alpha * steps) / (2.0 * sigma * sigma)
        eps = rdp + log_inv_delta / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps
            best_order = alpha

    return best_eps, best_order

eps, order = rdp_epsilon_approx(1/10, 2, 50, 0.001)
print(f"eps≈{eps:.3f}, best_order={order}")