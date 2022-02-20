# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:09:39 2022

@author: Thomas Maynadié
"""

import numpy as np
from ..quaternion_helper import quaternion

class rotation_sim_quaternions():
    def __init__(self, init_attitude, w, tf, N):
        self.q = quaternion(*init_attitude)
        self.w = w
        
        self.q_traj = [list(self.q.get_coefficients())]
        self.t = [0]
        
        dt = tf/N
        
        qi = quaternion(*self.q.get_coefficients())
        
        for i in range(1,N):
            qi = self.__compute_attitude_estimate(w, qi, dt)
                
            self.q_traj.append(list(qi.get_coefficients()))
            self.t.append(self.t[i-1] + dt)
            
    def get_trajectory(self):
        return np.array(self.t), np.array(self.q_traj)
    
    def measurement(self, s_a, b_a, s_w, b_w, s_mag, n):
        w_meas_1 = self.w + np.random.normal(b_w, np.sqrt(s_w), 3)
        w_meas_2 = self.w + np.random.normal(b_w, np.sqrt(s_w), 3)
        
        q0, q1, q2, q3 = self.q_traj[n]
        
        B_enu = np.array([0, 1, 0])
        g = 1
        
        M = quaternion(*self.q_traj[n]).DCM().transpose() # compute ENU to RF rotation matrix
        B_rf = np.dot(M, B_enu)
        
        g_rf = np.array([-2*g* (q1 * q3 - q2 * q0), - 2 * g * (q2 * q3 + q1 * q0), - g * (1 - 2 * (q1 * q1 + q2 * q2))])
        
        B_meas_1 = B_rf + np.random.normal(0, np.sqrt(s_mag), 3)
        B_meas_2 = B_rf + np.random.normal(0, np.sqrt(s_mag), 3)

        a_meas_1 = g_rf + np.random.normal(b_a, np.sqrt(s_a), 3)
        a_meas_2 = g_rf + np.random.normal(b_a, np.sqrt(s_a), 3)
        
        return a_meas_1, a_meas_2, w_meas_1, w_meas_2, B_meas_1, B_meas_2
        
    def __compute_angular_rate_matrix(self, angular_rate):
        M = np.zeros((4, 4))
        
        M[0][1] = -angular_rate[0]
        M[0][2] = -angular_rate[1]
        M[0][3] = -angular_rate[2]
    
        M[1][2] = angular_rate[2]
        M[1][3] = -angular_rate[1]
    
        M[2][3] = angular_rate[0]
        
        M = M - np.transpose(M)
        
        return M
    
    def __compute_attitude_estimate(self, angular_rate, q, dt):                
        W = self.__compute_angular_rate_matrix(angular_rate)
        
        q_new = q.get_coefficients()
        q_new = q_new + dt/2 * np.dot(W, q_new)   
        
        normalization_factor = 1 / np.sqrt(np.sum(q_new**2))
        q_new = quaternion(*(q_new * normalization_factor))
        
        print(q_new.norm2())
        
        return q_new