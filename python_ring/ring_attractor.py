import numpy as np

class RingAttractor:
    def __init__(self, num_neurons, tau=1, dt=0.1, T=1000):
        self.num_neurons = num_neurons
        self.tau = tau
        self.dt = dt
        self.T = T

        #initialize the neurons
        self.theta = np.linspace(0, 2 * np.pi, num_neurons, endpoint=False)
        self.f = np.zeros(num_neurons) #initial activity profile
        self.mu_init = np.pi/2 #initial bump center, we'll say it's always at the front of the fly

    def von_mises(self, amplitude, kappa, theta, mu): #Their implementation of the function (Kim et al. 2017)
        if kappa > 5:
            return amplitude * np.exp(kappa * (np.cos(theta - mu) - 1))
        else:
            return amplitude * (np.exp(kappa * (np.cos(theta - mu) + 1)) - 1) / (np.exp(2 * kappa) - 1)

    def initial_bump(self, amplitude, kappa, center=None):
        """Initializes the activity profile via the von Mises function"""
        if center is None:
            center = self.mu_init
        self.f = self.von_mises(self, amplitude, kappa, self.theta, center)


class LocalModel(RingAttractor):
    def __init__(self, num_neurons, alpha=3, beta=20, D=0.1):
        super().__init__(num_neurons=num_neurons)
        self.alpha = alpha
        self.beta = beta
        self.D = D
        self.kernel = self._connectivity_kernel()
    
    def _connectivity_kernel(self):
        """This generates the connectivity for the local model"""
        kernel = np.zeros(self.num_neurons) #same size as num_neurons
        for neuron_index in range(self.num_neurons):
            d = min(neuron_index, self.num_neurons - neuron_index)
            if d == 1:
                kernel[neuron_index] = self.D
        return kernel


class GlobalModel(RingAttractor):
    def __init__(self, num_neurons, J0=-8, J1=6):
        super().__init__(num_neurons=num_neurons)
        self.J0 = J0
        self.J1 = J1

        self.kernel = self._connectivity_kernel()
    
    def _connectivity_kernel(self): #TODO
        """This generates the connectivity for the global model"""
        kernel = np.zeros(self.num_neurons) #same size as num_neurons
        pass


