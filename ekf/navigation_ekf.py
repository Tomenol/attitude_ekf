# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 09:10:49 2022

@author: Thomas Maynadié
"""

import numpy as np

from helpers.quaternion_helper import quaternion
from helpers.simulation.rotation_simulation import rotation_sim_quaternions

class NavigationEKF:
    def __init__(self, initial_attitude, initial_position, gyroscope_std_deviation, accelerometer_std_deviation, accelerometer_bias, magnetometer_std_deviation, magnetometer_bias, gps_pos_std_deviation, gps_vel_std_deviation):
        # measurement errors
        self.gyroscope_std_deviation = gyroscope_std_deviation

        self.accelerometer_std_deviation = accelerometer_std_deviation
        self.accelerometer_bias = accelerometer_bias
        
        self.gps_pos_std_deviation = gps_pos_std_deviation
        self.gps_vel_std_deviation = gps_vel_std_deviation
        
        self.magnetometer_std_deviation = magnetometer_std_deviation
        self.magnetometer_bias = magnetometer_bias
        
        # state vector x = [q0, q1, q2, q3, bwx, bwy, bwz].T
        # EKF parameters
        self.state_space_dim = 7
        self.measurement_space_dim = 6
        
        self.P = np.diag([*np.ones(4), *np.ones(3)])*.001
        self.x = np.zeros(self.state_space_dim)
        
        self.initial_quaternion = quaternion(*initial_attitude)
        self.x[0:4] = self.initial_quaternion.get_coefficients()

        self.R = np.diag([*self.magnetometer_std_deviation, *self.accelerometer_std_deviation])**2    
        
        self.m_declination = np.pi/2
        
        print("Navigation EKF initialized")
        
    def filter_step(self, input_vector, measurement_vector, dt):    
        # predict
        x_prediction, normalization_factor = self.state_transition_model_function(self.x, input_vector, dt)
        
        # compute process noise matrix
        L = self.state_transition_model_function_jacobian_inputs(x_prediction, dt, normalization_factor)
        quaternion_integration_process_noise_matrix = np.dot(L, np.dot(np.diag(self.gyroscope_std_deviation**2), L.transpose()))
        
        Q = np.diag([0, 0, 0, 0 ,*self.gyroscope_std_deviation**2])
        Q[0:4,0:4] = quaternion_integration_process_noise_matrix
        
        F = self.state_transition_model_function_jacobian(self.x, input_vector, normalization_factor, dt)
        P_prediction = np.dot(F, np.dot(self.P, F.transpose())) + Q
        
        # update
        H = self.prediction_measurement_vector_jacobian_gps(x_prediction)
        y = measurement_vector - self.prediction_measurement_vector_gps(x_prediction)
        
        S = np.dot(H, np.dot(P_prediction, H.transpose())) + self.R
        K = np.dot(P_prediction, np.dot(H.transpose(), np.linalg.inv(S)))
        
        self.x = x_prediction + np.dot(K, y)
        
        # normalize quaternion
        self.x[0:4] = self.x[0:4] / np.sqrt(np.sum(self.x[0:4]**2))
        
        self.P = np.dot((np.eye(self.state_space_dim) - np.dot(K, H)), P_prediction)
        self.P = (self.P + self.P.transpose())/2
        
        return self.x
    
    def compute_angular_rate_matrix(self, angular_rate):
        M = np.zeros((4, 4))
        
        M[0][1] = -angular_rate[0]
        M[0][2] = -angular_rate[1]
        M[0][3] = -angular_rate[2]
    
        M[1][2] = angular_rate[2]
        M[1][3] = -angular_rate[1]
    
        M[2][3] = angular_rate[0]
        
        M = M - np.transpose(M)
        
        return M
    
    def state_transition_model_function(self, x, u, dt):
        # enu frame
        q0 = x[0]
        q1 = x[1]
        q2 = x[2]
        q3 = x[3]
        
        # rocket frame
        wx = u[0]
        wy = u[1]
        wz = u[2]
        
        wbx = x[4]
        wby = x[5]
        wbz = x[6]
        
        x_new = x.copy()
            
        x_new[0] += (-q1 * (wx - wbx) - q2 * (wy - wby) - q3 * (wz - wbz)) * dt/2
        x_new[1] += ( q0 * (wx - wbx) - q3 * (wy - wby) + q2 * (wz - wbz)) * dt/2
        x_new[2] += ( q3 * (wx - wbx) + q0 * (wy - wby) - q1 * (wz - wbz)) * dt/2
        x_new[3] += (-q2 * (wx - wbx) + q1 * (wy - wby) + q0 * (wz - wbz)) * dt/2
        
        normalization_factor = 1 / np.sqrt(np.sum(x_new[0:4]**2))
        x_new[0:4] = x_new[0:4] * normalization_factor
        
        return x_new, normalization_factor
    
    def state_transition_model_function_jacobian(self, x, u, normalization_factor, dt):
        F = np.eye(self.state_space_dim)
        
        q0 = x[0]
        q1 = x[1]
        q2 = x[2]
        q3 = x[3]
        
        wx = u[0]
        wy = u[1]
        wz = u[2]
        
        wbx = x[4]
        wby = x[5]
        wbz = x[6]     
                                    
        # q0 = (q0 + (-q1 * (wx - wbx) - q2 * (wy - wby) - q3 * (wz - wbz) * dt/2) * normalization_factor
        F[0][0] = normalization_factor
        F[0][1] = -normalization_factor * (wx - wbx) * dt/2
        F[0][2] = -normalization_factor * (wy - wby) * dt/2
        F[0][3] = -normalization_factor * (wz - wbz) * dt/2
        
        F[0][4] = normalization_factor * q1 * dt/2
        F[0][5] = normalization_factor * q2 * dt/2
        F[0][6] = normalization_factor * q3 * dt/2
        
        # q1 = (q1 + (q0 * (wx - wbx) - q3 * (wy - wby) + q2 * (wz - wbz)) * dt/2) * normalization_factor
        F[1][0] = normalization_factor * (wx - wbx) * dt/2
        F[1][1] = normalization_factor
        F[1][2] = normalization_factor * (wz - wbz) * dt/2
        F[1][3] = -normalization_factor * (wy - wby) * dt/2
        
        F[1][4] = -normalization_factor * q0 * dt/2
        F[1][5] = normalization_factor * q3 * dt/2
        F[1][6] = -normalization_factor * q2 * dt/2
        
        # q2 = (q2 + (q3 * (wx - wbx) + q0 * (wy - wby) - q1 * (wz - wbz)) * dt/2) * normalization_factor
        F[2][0] = normalization_factor * (wy - wby) * dt/2
        F[2][1] = -normalization_factor * (wz - wbz) * dt/2
        F[2][2] = normalization_factor
        F[2][3] = normalization_factor * (wx - wbx) * dt/2
        
        F[2][4] = -normalization_factor * q3 * dt/2
        F[2][5] = -normalization_factor * q0 * dt/2
        F[2][6] = normalization_factor * q1 * dt/2
    
        # q3 = (q3 + (-q2 * (wx - wbx) + q1 * (wy - wby) + q0 * (wz - wbz)) * dt/2) * normalization_factor
        F[3][0] = normalization_factor * (wz - wbz) * dt/2
        F[3][1] = normalization_factor * (wy - wby) * dt/2
        F[3][2] = -normalization_factor * (wx - wbx) * dt/2
        F[3][3] = normalization_factor
        
        F[3][4] = normalization_factor * q2 * dt/2
        F[3][5] = -normalization_factor * q1 * dt/2
        F[3][6] = -normalization_factor * q0 * dt/2
        
        return F
    
    def prediction_measurement_vector_gps(self, x):
        q0, q1, q2, q3 = x[0:4]

        sin_m = np.sin(self.m_declination)
        cos_m = np.cos(self.m_declination)
        
        mx = 2*sin_m*(q0*q3 + q1*q2)-cos_m*(2*q2**2 + 2*q3**2 - 1)
        my = -2*cos_m*(q0*q3 - q1*q2)-sin_m*(2*q1**2 + 2*q3**2 - 1)
        mz = 2*cos_m*(q0*q2 + q1*q3)-2*sin_m*(q0*q1 - q2*q3)
        
        g = 1
        
        ax = - 2 * g * (q1 * q3 - q2 * q0)
        ay = - 2 * g * (q2 * q3 + q1 * q0)
        az = - g* (1 - 2 * (q1 * q1 + q2 * q2))
        
        return np.array([mx, my, mz, ax, ay, az])
    
    def prediction_measurement_vector_jacobian_gps(self, x):
        H = np.zeros((self.measurement_space_dim, self.state_space_dim))
                
        q0, q1, q2, q3 = x[0:4]
        
        # magnetometer measurements (3 -> 5)
        sin_m = np.sin(self.m_declination)
        cos_m = np.cos(self.m_declination)
        
        g = 1
        
        # (q0**2 + q1**2 - q2**2 - q3**2)*mx_enu + 2*(q1*q2+q0*q3)*my_enu + 2*(q1*q3-q0*q2)*mz_enu
        H[0][0] = 2*q3*sin_m
        H[0][1] = 2*q2*sin_m
        H[0][2] = 2*(q1*sin_m - 2*q2*cos_m)
        H[0][3] = 2*(q0*sin_m - 2*q3*cos_m)
        
        #my = 2*(q1*q2-q0*q3)*mx_enu + (q0**2 - q1**2 + q2**2 - q3**2)*my_enu + 2*(q2*q3+q0*q1)*mz_enu
        H[1][0] = -2*q3*cos_m
        H[1][1] = 2*(q2*cos_m - 2*q1*sin_m)
        H[1][2] = 2*q1*cos_m
        H[1][3] = -2*(q0*cos_m + 2*q3*sin_m)
        
        #mz = 2*(q1*q3+q0*q2)*mx_enu + 2*(q2*q3-q0*q1)*my_enu + (q0**2 - q1**2 - q2**2 + q3**2)*mz_enu
        H[2][0] = 2*(q2*cos_m - q1*sin_m)
        H[2][1] = 2*(q3*cos_m - q0*sin_m)
        H[2][2] = 2*(q0*cos_m + q3*sin_m)
        H[2][3] = 2*(q1*cos_m + q2*sin_m)
        
        # ax = - 2 * g * (q1 * q3 - q2 * q0)
        H[3][0] = 2*g*q2
        H[3][1] = -2*g*q3
        H[3][2] = 2*g*q0
        H[3][3] = -2*g*q1
        
        # ay = - 2 * g * (q2 * q3 + q1 * q0)
        H[4][0] = -2*g*q1
        H[4][1] = -2*g*q0
        H[4][2] = -2*g*q3
        H[4][3] = -2*g*q2
        
        # az = - g* (1 - 2 * (q1 * q1 + q2 * q2))
        H[5][1] = 4*g*q1
        H[5][2] = 4*g*q2
        
      
        return H
    
    def state_transition_model_function_jacobian_inputs(self, x, dt, normalization_factor):
        F = np.eye(4, 3)
        
        q0 = x[0]
        q1 = x[1]
        q2 = x[2]
        q3 = x[3]
                                    
        # q0 = (q0 + (-q1 * (wx - wbx) - q2 * (wy - wby) - q3 * (wz - wbz) * dt/2) * normalization_factor
        F[0][0] = -normalization_factor * q1 * dt/2
        F[0][1] = -normalization_factor * q2 * dt/2
        F[0][2] = -normalization_factor * q3 * dt/2
        
        # q1 = (q1 + (q0 * (wx - wbx) - q3 * (wy - wby) + q2 * (wz - wbz)) * dt/2) * normalization_factor
        F[1][0] = normalization_factor * q0 * dt/2
        F[1][1] = -normalization_factor * q3 * dt/2
        F[1][2] = normalization_factor * q2 * dt/2
        
        # q2 = (q2 + (q3 * (wx - wbx) + q0 * (wy - wby) - q1 * (wz - wbz)) * dt/2) * normalization_factor
        F[2][0] = normalization_factor * q3 * dt/2
        F[2][1] = normalization_factor * q0 * dt/2
        F[2][2] = -normalization_factor * q1 * dt/2
    
        # q3 = (q3 + (-q2 * (wx - wbx) + q1 * (wy - wby) + q0 * (wz - wbz)) * dt/2) * normalization_factor
        F[3][0] = -normalization_factor * q2 * dt/2
        F[3][1] = normalization_factor * q1 * dt/2
        F[3][2] = normalization_factor * q0 * dt/2
      
        return F


