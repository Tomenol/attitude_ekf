# Attitude Extended Kalman Filter
The purpose of this code is to compute the optimal state estimate of a rigid system in rotation in the ENU (East-North-Up) reference frame. It uses an extended Kalman filter to fuse the measurements from two IMUs (accelerometer + gyroscope) and two magnetometers. 

The script is complemented with a script which simulates the system's rotation, computes the theoretical measurements and adds random measurement noise. Then the main script gathers the measurements and computes the optimal state vector estimate using the following algorithm :

![image](https://user-images.githubusercontent.com/54234406/154864160-3ebc088a-677d-4e04-bd7d-3c4a7cf17f00.png)

The EKF algorithm uses the following state vector :

![image](https://user-images.githubusercontent.com/54234406/154864400-b1cf3e39-0c7d-4a4c-8647-1013752818be.png)

# results :
By running the simulation, we can obtain the following results :

![image](https://user-images.githubusercontent.com/54234406/155009076-cac46f25-6842-42f7-86d7-a83f6a9b33d0.png)

Legend : 
red curve -> theoretical value

blue curve -> optimal EKF state estimate


