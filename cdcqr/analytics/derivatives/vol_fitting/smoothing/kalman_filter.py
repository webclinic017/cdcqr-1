import numpy as np
import matplotlib.pyplot as plt


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
    n = 7
    m = np.shape(h_mat)[0]
    
    filter.x = x_prev
    filter.P = sigma1*np.eye(n)
    filter.A = np.eye(n)
    filter.H = h_mat
    filter.HT = filter.H.T
    filter.Q = sigma2*np.eye(n)
        
    # Predict State Forward
    x_p = filter.A.dot(filter.x)
    # Predict Covariance Forward
    P_p = filter.A.dot(filter.P).dot(filter.A.T) + filter.Q
    # Compute Kalman Gain
    S = filter.H.dot(P_p).dot(filter.HT)
    K = P_p.dot(filter.HT)*(1/S)
    # Estimate State
    residual = z - filter.H.dot(x_p)
    filter.x = x_p + K*residual
    # Estimate Covariance
    filter.P = P_p - K.dot(filter.H).dot(P_p)
    return filter.x


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