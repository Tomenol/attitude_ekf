# Attitude Extended Kalman Filter
The purpose of this code is to compute the optimal state estimate of a rigid system in rotation in the ENU (East-North-Up) reference frame. It uses an extended Kalman filter to fuse the measurements from two IMUs (accelerometer + gyroscope) and two magnetometers. 

The script is complemented with a script which simulates the system's rotation, computes the theoretical measurements and adds random measurement noise. Then the main script gathers the measurements and computes the optimal state vector estimate using the following algorithm :

![image](https://user-images.githubusercontent.com/54234406/154864160-3ebc088a-677d-4e04-bd7d-3c4a7cf17f00.png)

\theta_0
