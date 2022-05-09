import tensorflow as tf
import matplotlib.pyplot as plt
import timeit

a = 0.2
b = 0.04
c = 0.4
d = 0.05

d_esa = lambda esa, hoshokusha: a*esa - b*esa*hoshokusha
d_hoshokusha = lambda esa, hoshokusha: -c*hoshokusha + d*esa*hoshokusha

def runge_kutta(f, x, y, dt):
    k1 = dt*f(x,y)
    k2 = dt*f(x+0.5*k1, y+0.5*k1)
    k3 = dt*f(x+0.5*k2, y+0.5*k2)
    k4 = dt*f(x+k3, y+k3)
    return (k1+2.0*k2+2.0*k3+k4)/6.0

@tf.function
def lotka_volterra(t_max, t_num, esa_init, hoshokusha_init):
    esa_result = []
    hoshokusha_result = []
    t_result = []

    t = 0.0
    esa = esa_init
    hoshokusha = hoshokusha_init

    esa_result.append(esa)
    hoshokusha_result.append(hoshokusha)
    t_result.append(t)

    dt = t_max / t_num

    while t < t_max:
        t += dt
        esa += runge_kutta(d_esa, esa, hoshokusha, dt)
        hoshokusha += runge_kutta(d_hoshokusha, esa, hoshokusha, dt)

        esa_result.append(esa)
        hoshokusha_result.append(hoshokusha)
        t_result.append(t)

    return esa_result, hoshokusha_result, t_result

# warm up！！！！！！
esa_result, hoshokusha_result, t_result = lotka_volterra(100.0, 2000, 10, 1)
print(timeit.timeit(lambda: lotka_volterra(100000.0, 2000, 10, 1), number=1))

plt.plot(t_result, esa_result)
plt.plot(t_result, hoshokusha_result)
plt.show()