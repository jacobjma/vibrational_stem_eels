import numpy as np

def closest_power_of_2(n):
    return 2 ** int(np.ceil(np.log(n) / np.log(2)))

def autocorrelation_1d(data):
    n = len(data)
    X = np.zeros(2 * closest_power_of_2(n))
    X[:n] = data
    F = np.fft.fft(X)
    result = np.fft.ifft(F * F.conj())[:n].real / (n - np.arange(n))
    return result[:n]

def autocorrelation(X):
    X = np.asarray(X)
    if X.ndim==1:
        return autocorrelation_1d(X)
    
    else:
        result = autocorrelation_1d(X[:,0])
        for j in range(1, X.shape[1]):
            result += autocorrelation_1d(X[:,j])
        return result