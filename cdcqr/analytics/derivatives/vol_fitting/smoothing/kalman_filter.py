import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


def getMeasurement(updateNumber):
    if updateNumber == 1:
        getMeasurement.currentPosition = 0
        getMeasurement.currentVelocity = 60 # m/s
    dt = 0.1
    w = 8 * np.random.randn(1)
    v = 8 * np.random.randn(1)
    z = getMeasurement.currentPosition + getMeasurement.currentVelocity*dt + v
    getMeasurement.currentPosition = z - v
    getMeasurement.currentVelocity = 60 + w
    return [z, getMeasurement.currentPosition, getMeasurement.currentVelocity]


def simple_kalman_filter(z, x_prev, sigma1, h_mat, sigma2):
    n = 7  # dimension of signal process, p0, p1, ... p5
    m = np.shape(h_mat)[0] # dimension of obs process 31, 31 strikes
    filter_ = simple_kalman_filter
    filter_.x = x_prev           # prev signal states
    filter_.P = sigma1*np.eye(n) # signal states cov
    filter_.A = np.eye(n)        # signal transition matrix
    filter_.H = h_mat            # signal2obs mat
    filter_.HT = filter_.H.T      
    filter_.Q = sigma1*np.eye(n) # signal process noise cov
    filter_.R = sigma2*np.eye(m) # obs process noise cov
    
        
    # Predict State Forward
    x_p = filter_.A.dot(filter_.x)
    # Predict Covariance Forward
    P_p = filter_.A.dot(filter_.P).dot(filter_.A.T) + filter_.Q
    # Compute Kalman Gain
    S = filter_.H.dot(P_p).dot(filter_.HT) + filter_.R
    K = P_p.dot(filter_.HT).dot(inv(S))
    # Estimate State
    residual = z - filter_.H.dot(x_p)
    filter_.x = x_p + K.dot(residual)
    # Estimate Covariance
    # filter_.P = P_p - K.dot(filter_.H).dot(P_p)
    return filter_.x


def testFilter():
    dt = 0.1
    t = np.linspace(0, 10, num=300)
    numOfMeasurements = len(t)
    measTime = []
    measPos = []
    measDifPos = []
    estDifPos = []
    estPos = []
    estVel = []
    posBound3Sigma = []
    for k in range(1,numOfMeasurements):
        z = getMeasurement(k)
        # Call Filter and return new State
        f = filter(z[0], k)
        # Save off that state so that it could be plotted
        measTime.append(k)
        measPos.append(z[0])
        measDifPos.append(z[0]-z[1])
        estDifPos.append(f[0]-z[1])
        estPos.append(f[0])
        estVel.append(f[1])
        posVar = f[2]
        posBound3Sigma.append(3*np.sqrt(posVar[0][0]))
    return [measTime, measPos, estPos, estVel, measDifPos, estDifPos, posBound3Sigma]


if __name__ == '__main__':
    t = testFilter()
    plot1 = plt.figure(1)
    plt.scatter(t[0], t[1])
    plt.plot(t[0], t[2])
    plt.ylabel('Position')
    plt.xlabel('Time')
    plt.grid(True)
    plot2 = plt.figure(2)
    plt.plot(t[0], t[3])
    plt.ylabel('Velocity (meters/seconds)')
    plt.xlabel('Update Number')
    plt.title('Velocity Estimate On Each Measurement Update \n', fontweight="bold")
    plt.legend(['Estimate'])
    plt.grid(True)
    plot3 = plt.figure(3)
    plt.scatter(t[0], t[4], color = 'red')
    plt.plot(t[0], t[5])
    plt.legend(['Estimate', 'Measurement'])
    plt.title('Position Errors On Each Measurement Update \n', fontweight="bold")
    #plt.plot(t[0], t[6])
    plt.ylabel('Position Error (meters)')
    plt.xlabel('Update Number')
    plt.grid(True)
    plt.xlim([0, 300])
    plt.show()