import numpy as np
import scipy.fft
import scipy.special
import sys
import matplotlib.pyplot as plt

class Parasw:
    def __init__(self):
        self.nbpts = 256
        self.dx = 2 * np.pi / self.nbpts
        self.dt = 0.1
        self.offcon = -0.1
        self.w1 = 0.5
        self.a1 = 1.0
        self.w2 = 1.0
        self.a2 = 0.5
        self.offin = 0.2
        self.ain = 3.0  # Increased input amplitude
        self.win = 0.5
        self.D = 0.1
        self.alpha = 3.0
        self.beta = 20.0

class FFT:
    def __init__(self, n):
        self.n = n
        self.connect = np.zeros(n)
        self.fftcon = np.zeros(n, dtype=complex)

def vonmises(x, mu, kappa):
    return np.exp(kappa * (np.cos(x - mu) - 1))

def init_connect(con, par):
    k1 = 1.0 / (par.w1 ** 2)
    k2 = 1.0 / (par.w2 ** 2)
    for i in range(par.nbpts):
        x = 2 * np.pi * i / par.nbpts
        con[i] = (par.offcon +
                  par.a1 * vonmises(x, 0.0, k1) -
                  par.a2 * vonmises(x, 0.0, k2))

def deriv_fft(array, input_signal, par, fft_struct):
    fft_array = scipy.fft.fft(array)
    convolved = scipy.fft.ifft(fft_array * fft_struct.fftcon).real
    convolved /= par.nbpts
    darray = convolved + 1 + input_signal
    darray = np.maximum(darray, 0)
    darray -= array
    return darray

def dynamics_rk4step_fft(curstate, input_signal, par, fft_struct):
    dt = par.dt
    k1 = deriv_fft(curstate, input_signal, par, fft_struct)
    k2 = deriv_fft(curstate + dt * k1 / 2, input_signal, par, fft_struct)
    k3 = deriv_fft(curstate + dt * k2 / 2, input_signal, par, fft_struct)
    k4 = deriv_fft(curstate + dt * k3, input_signal, par, fft_struct)
    curstate += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def init_sw(curstate, input_signal, par, fft_struct):
    for _ in range(int(20.0 / par.dt)):
        dynamics_rk4step_fft(curstate, input_signal, par, fft_struct)

def jump_vs_flow_input(curstate, input_signal, par, fft_struct, model, jump_dist):
    dim = par.nbpts
    
    with open("simulation_results.txt", "w") as result_file:
        for width in np.linspace(0.1, np.pi, 20):
            for amp in np.linspace(0.5, 5.0, 20):
                # Reset initial conditions
                curstate[:] = 0.1
                
                # Create input at different positions
                initial_pos = 0
                jump_pos = np.deg2rad(jump_dist)
                
                # Initial stabilization input
                kin_init = 1.0 / (width ** 2)
                for i in range(dim):
                    x = 2 * np.pi * i / dim
                    input_signal[i] = amp * vonmises(x, initial_pos, kin_init)
                
                # Stabilize initial state
                init_sw(curstate, input_signal, par, fft_struct)
                initial_max_pos = np.argmax(curstate)
                
                # Apply jump input
                for i in range(dim):
                    x = 2 * np.pi * i / dim
                    input_signal[i] = amp * vonmises(x, jump_pos, kin_init)
                
                # Run dynamics
                for _ in range(int(20.0 / par.dt)):
                    dynamics_rk4step_fft(curstate, input_signal, par, fft_struct)
                
                # Record result
                final_max_pos = np.argmax(curstate)
                result = f"Width: {width:.3f}, Amp: {amp:.3f}, InitialMaxPos: {initial_max_pos}, FinalMaxPos: {final_max_pos}"
                print(result)
                result_file.write(result + "\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python ring_attractor.py [delta|cosine] jump_distance")
        return

    model = sys.argv[1]
    jump_dist = float(sys.argv[2])

    par = Parasw()
    fft_struct = FFT(par.nbpts)

    # Initialize connectivity
    con = fft_struct.connect
    init_connect(con, par)

    # Plot connectivity
    plt.figure()
    plt.plot(con)
    plt.title('Connectivity Profile')
    plt.xlabel('Neuron Index')
    plt.ylabel('Connectivity Strength')
    plt.savefig('connectivity_profile.png')
    plt.close()

    # Compute FFT of connectivity
    fft_struct.fftcon = scipy.fft.fft(con)

    # Initialize state and input
    curstate = np.zeros(par.nbpts)
    input_signal = np.zeros(par.nbpts)

    # Run simulation
    jump_vs_flow_input(curstate, input_signal, par, fft_struct, model, jump_dist)

    # Plot final state
    plt.figure()
    plt.plot(curstate)
    plt.xlabel('Neuron Index')
    plt.ylabel('Activity')
    plt.title('Final State after Simulation')
    plt.savefig('final_state.png')
    plt.close()

if __name__ == "__main__":
    main()