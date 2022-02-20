# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:39:21 2022

@author: Thomas Maynadié
"""
import numpy as np
import matplotlib.pyplot as plt

import prefilters.imu_prefilters as imu_prefilters
import ekf.navigation_ekf as navigation_ekf

import helpers.simulation.rotation_simulation as rotation_simulation

from helpers.quaternion_helper import quaternion


def main_routine():
    # initialization
    # get sensor parameters from the first 30 still measurements
   
    # --------------------------------------------
    #               sensor definition
    # --------------------------------------------
    
    # 1st MARG sensor
    # gyroscope
    gyroscope_std_deviation_1 = np.array([0.1, 0.12, 0.098])
    gyroscope_bias_1 = np.array([0.2, 0.15, 0.12])
    
    # accelerometer
    accelerometer_std_deviation_1 = np.array([0.072, 0.078, 0.069])
    accelerometer_bias_1 = np.array([0.03, 0.01, 0.02])
    
    # magnetometer
    magnetometer_std_deviation_1 = np.array([0.2, 0.21, 0.19])
    magnetometer_bias_1 = np.array([0.11, 0.15, 0.12])
    
    # 2nd MARG sensor
    # gyroscope
    gyroscope_std_deviation_2 = np.array([0.1, 0.12, 0.098])
    gyroscope_bias_2 = np.array([0.2, 0.15, 0.12])
    
    # accelerometer
    accelerometer_std_deviation_2 = np.array([0.072, 0.078, 0.069])
    accelerometer_bias_2 = np.array([0.03, 0.01, 0.02])
    
    # magnetometer
    magnetometer_std_deviation_2 = np.array([0.2, 0.21, 0.19])
    magnetometer_bias_2 = np.array([0.11, 0.15, 0.12])
    
    # --------------------------------------------
    #            Component defintion
    # --------------------------------------------
    
    # Gyroscope pre-filter
    gyroscope_prefilter = imu_prefilters.gyroscope_preprocessor(gyroscope_std_deviation_1, gyroscope_bias_1, gyroscope_std_deviation_2, gyroscope_bias_2)
    
    # Magnetometer + Accelerometer preprocessing unit
    accelerometer_prepreprocessor = imu_prefilters.imu_sensor_preprocessor(accelerometer_std_deviation_1, accelerometer_bias_1, accelerometer_std_deviation_2, accelerometer_bias_2)
    magnetometer_prepreprocessor = imu_prefilters.imu_sensor_preprocessor(magnetometer_std_deviation_1, magnetometer_bias_1, magnetometer_std_deviation_2, magnetometer_bias_2)
    
    accelerometer_processed_std_deviation = accelerometer_prepreprocessor.get_std_deviation()
    accelerometer_processed_bias = accelerometer_prepreprocessor.get_bias()
    
    gyroscope_processed_std_deviation = gyroscope_prefilter.get_std_deviation()
    gyroscope_processed_bias = gyroscope_prefilter.get_bias()
    
    magnetometer_processed_std_deviation = magnetometer_prepreprocessor.get_std_deviation()
    magnetometer_processed_bias = magnetometer_prepreprocessor.get_bias()
    
    # GPS preprocessor
    gps_pos_std_deviation = np.zeros(3)
    gps_vel_std_deviation = np.zeros(3)
    # TODO
    
    # --------------------------------------------
    #               Constants
    # --------------------------------------------
    
    # All global quantities are given in the ENU reference frame
    global_gravity_vector = np.array([0, 0, -1])    
    global_magnetic_vector = np.array([0, 1, 0])
    
    # --------------------------------------------
    #               Rocket geometry
    # --------------------------------------------
    
    # All local quantities are given in the Rocket Frame (RF) reference frame (x : main rocket axis / y, z : arbitrarly chosen with respect to the navigation module's geometry)
    
    # Navigation module position
    r_rocket_cog_to_nav_ref = np.array([0.55, 0, 0]) # tbd with structural team
    
    # MARGs position
    r_nav_ref_to_marg1 = np.array([0.1, -1, 0]) # tbd with structural team
    r_nav_ref_to_marg2 = np.array([0.1, -1, 0]) # tbd with structural team
    
    # --------------------------------------------
    #             Other variables
    # --------------------------------------------
    iteration_nb = 1000
    
    t0 = 0
    tf = 100
    dt = (tf - t0)/iteration_nb
    
    # simulation
    initial_attitude = np.array([0, -90, 0])
    initial_position = np.array([0, 0, 0])
    
    angular_rate = np.array([2, 0.0, 0.0])
    simulation = rotation_simulation.rotation_sim_quaternions(initial_attitude, angular_rate, tf, iteration_nb)
    
    # results
    time_array, theoretical_trajectory = simulation.get_trajectory()
    estimated_trajectory = [[*quaternion(*initial_attitude).get_coefficients(), 0, 0, 0]]
    time_array = np.linspace(t0, tf, iteration_nb)
    
    w_m = [angular_rate]
    
    # --------------------------------------------
    #             Kalman Variables
    # --------------------------------------------
    navigation_state_estimator = navigation_ekf.NavigationEKF(initial_attitude, initial_position, gyroscope_processed_std_deviation, accelerometer_processed_std_deviation, accelerometer_processed_bias, magnetometer_processed_std_deviation, magnetometer_processed_bias, gps_pos_std_deviation, gps_vel_std_deviation)

    # --------------------------------------------
    #               MAIN LOOP
    # --------------------------------------------
    for i in range(1, iteration_nb):
        # get measurement 
        raw_accelerometer_measurement_1, raw_accelerometer_measurement_2, raw_gyroscope_measurement_1, raw_gyroscope_measurement_2, raw_magnetometer_measurement_1, raw_magnetometer_measurement_2 = simulation.measurement(accelerometer_processed_std_deviation, accelerometer_processed_bias, gyroscope_processed_std_deviation, gyroscope_processed_bias, magnetometer_processed_std_deviation, i)
        
        # preprocess measurements
        processed_accelerometer_measurement = accelerometer_prepreprocessor.process_measurements(raw_accelerometer_measurement_1, raw_accelerometer_measurement_2)
        processed_gyroscope_measurement, estimated_angular_acceleration = gyroscope_prefilter.process_measurements(raw_gyroscope_measurement_1, raw_gyroscope_measurement_2, dt)
        processed_magnetometer_measurement = magnetometer_prepreprocessor.process_measurements(raw_magnetometer_measurement_1, raw_magnetometer_measurement_2)
        
        w_m.append(processed_gyroscope_measurement)
        
        # Kalman step
        measurement_vector = np.array([*processed_magnetometer_measurement, *processed_accelerometer_measurement])
        input_vector = np.array([*processed_gyroscope_measurement])
        
        estimated_trajectory.append(navigation_state_estimator.filter_step(input_vector, measurement_vector, dt))
    
    # --------------------------------------------
    #               POST PROCESSING
    # --------------------------------------------
    estimated_trajectory = np.array(estimated_trajectory).transpose()
    theoretical_trajectory = np.array(theoretical_trajectory).transpose()
    w_m = np.array(w_m).transpose()
    
    fig = plt.figure(dpi=300)
    subplots = fig.subplots(7, 1)
    
    y_labels = ["q0", "q1", "q2", "q3", "b_wx", "b_wy", "b_wz"]
    
    for i in range(4):
        subplots[i].plot(time_array, theoretical_trajectory[i], "r")
        subplots[i].plot(time_array, estimated_trajectory[i], "b")
        
        subplots[i].set_xlabel("t [s]")
        subplots[i].set_ylabel(y_labels[i])
        subplots[i].grid()
        
    for i in range(4,7):
        subplots[i].plot(time_array, np.ones(iteration_nb)*gyroscope_processed_bias[i-4], "r")
        subplots[i].plot(time_array, estimated_trajectory[i], "b")
        
        subplots[i].set_xlabel("t [s]")
        subplots[i].set_ylabel(y_labels[i])
        subplots[i].grid()

    print("done")
    
if __name__ == '__main__':
    main_routine()