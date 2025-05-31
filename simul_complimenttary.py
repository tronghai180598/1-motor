# Kalman filter đã làm việc tốt trong tất ca mọi trường hợp
# OK ok có nghĩa là hệ số K là tự động tính.
# Lấy ra 3 trường hợp:
# 1. K11 = 0.05(Hệ số đầu tiên) (khi Q =0.0004 ) ĐÂY LÀ TRUỜNG HỢP TỐI ƯU NHẤT
# 2. K11 = 0.17 (khi tăng Q đi 10 lần Q =0.004 ) ĐÂY LÀ TRUỜNG HỢP CÓ NHIỄU NHIỀU HƠN
# 3. K11 = 0.012   (khi giảm Q lên 10 lần Q =0.00004 ) ĐÂY LÀ TRUỜNG HỢP CÓ NHIỄU ÍT HƠN NHƯNG FILTER LỪA DỐI HƠN.

import random
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

deg2rad = pi / 180.0


class ControlAngular:
    def __init__(self, P_angular, D_angular, theta_des):
        self.P_angular = P_angular
        self.D_angular = D_angular
        self.theta_des = theta_des

    def update_angle(self, theta, theta_dot, dt):
        error = self.theta_des - theta
        P = self.P_angular * error
        D = self.D_angular * theta_dot
        theta_dot_des = P - D
        return theta_dot_des


class ControlRate:
    def __init__(self, P_rate, D_rate, theta_dot_des):
        self.P_rate = P_rate
        self.D_rate = D_rate
        self.theta_dot_des = theta_dot_des

    def update_rate(self, theta_dot, theta_ddot, dt):
        error = self.theta_dot_des - theta_dot
        P = self.P_rate * error
        D = self.D_rate * theta_ddot
        d_u = P - D
        return d_u

class FirstOrder:
    def __init__(self, Ki, dt):
        self.Ki = Ki
        self.dt = dt
        self.y = 0

    def step(self, x):
        self.y = (x * self.dt + self.Ki * self.y) / (self.dt + self.Ki)
        return self.y

class Integrator:
    def __init__(self, K, dt):
        self.K = K
        self.dt = dt
        self.y = 0

    def step(self, x):
        self.y += self.K * x * self.dt
        return self.y

class ComplementaryFilter:
    def __init__(self, dt, alpha):
        self.dt = dt
        self.alpha = alpha
        self.angle = 0.0  # Начальный угол
        self.velocity = 0.0  # Начальная угловая скорость

    def update(self, gyro_rate, accel_angle):
        # Интеграция гироскопа для получения угла
        self.velocity += gyro_rate * self.dt
        self.angle += self.velocity * self.dt
        # Коррекция с использованием акселерометра
        self.angle = self.alpha * (self.angle + gyro_rate * self.dt) + (1 - self.alpha) * accel_angle

        return self.angle, self.velocity

class Simulator:
    def __init__(self, Tend, dt):
        self.dt = dt
        self.Tend = Tend
        self.theta_dot_List = []
        self.theta_List = []

        self.theta_dot_comple_List = []
        self.theta_comple_List = []

        self.timeList = []

    def runSimulation(self):
        Kv = 32767 / 300
        Kf = 2
        Ki = 0.003 * 1000
        Tm = 25 / 1000
        g = 2

        # Adjusted gains for smaller dt
        Kdv = 1 / (Tm * Ki * g)
        Kpv = Kdv / (2 * Tm * g * Kv)
        Kpf = 1 / (Kf * g * 3.125 * Tm)
        Kdf = 1.25 * Tm * Kpf * Kf

        integ_theta_dot = Integrator(Kv, self.dt)
        integ_theta = Integrator(Kf, self.dt)
        integ_motor = Integrator(Ki, self.dt)
        angular = ControlAngular(Kpf, Kdf, 100)
        first_order = FirstOrder(Tm, self.dt)
        complement = ComplementaryFilter(self.dt, 0.98)

        time = 0
        u = 0  # Initial value for the control signal

        while time <= self.Tend:
            # Apply noise
            noise = 2.0 * np.sin(42 * time) + 0.2 * np.sin(242.7 * time + 60 * deg2rad)
            u += noise
            a = first_order.step(u)

            theta_dot = integ_theta_dot.step(a)
            theta_dot += random.uniform(-40, 40)
            theta = integ_theta.step(theta_dot)
            theta += random.uniform(-40, 40)
            com_angle, com_velocity = complement.update(theta_dot, theta)

            v_des = angular.update_angle(com_angle, com_velocity, self.dt)
            rate = ControlRate(Kpv, Kdv, v_des)
            d_u = rate.update_rate(theta_dot, a, self.dt)

            u = integ_motor.step(d_u)

            self.theta_dot_List.append(theta_dot)
            self.theta_List.append(theta)

            self.theta_dot_comple_List.append(theta_dot)
            self.theta_comple_List.append(com_angle)
            # self.theta_ddot_kalman_List.append(theta_ddot_kalman)

            self.timeList.append(time)
            time += self.dt

    def showPlots(self):
        plt.figure()
        font = {'family': 'serif', 'weight': 'normal', 'size': 12}
        plt.rc('font', **font)

        # plt.subplot(111)
        plt.plot(self.timeList, self.theta_dot_List, label="Vi", color='b')  # Синий
        plt.plot(self.timeList, self.theta_List, label="Fi", color='g')     # Красный
        plt.plot(self.timeList, self.theta_comple_List, label="cFi", color='m')  # Пурпурный

        plt.legend()
        plt.show()


dt = 0.001  # Simulation step size
Tend = 3  # End time for simulation

# Run the simulation
sim = Simulator(Tend, dt)
sim.runSimulation()
sim.showPlots()
