import numpy as np
import scipy.fft
import scipy.special
import sys
import matplotlib.pyplot as plt

# Define parameters and structures
class Parasw:
    def __init__(self):
        self.nbpts = 256  # Number of points (neurons)
        self.dx = 2 * np.pi / self.nbpts
        self.dt = 0.01  # Time step
        self.trelax = 10.0  # Relaxation time
        # Connectivity parameters
        self.offcon = -0.1
        self.w1 = 0.5
        self.a1 = 1.0
        self.w2 = 1.0
        self.a2 = 0.5
        # Input parameters
        self.offin = 0.1
        self.ain = 1.0
        self.win = 0.5
        # Additional parameters
        self.D = 0.1
        self.alpha = 3.0
        self.beta = 20.0
        self.J0 = -0.5
        self.J1 = 1.0
        self.conwidth = 0  # Will be calculated

# Define the FFT structure
class FFT:
    def __init__(self, n):
        self.n = n
        self.in_array = np.zeros(n)
        self.connect = np.zeros(n)
        self.fftcon = np.zeros(n, dtype=complex)

# Utility functions
def vonmises(x, mu, kappa):
    if kappa > 5:
        return np.exp(kappa * (np.cos(x - mu) - 1))
    else:
        numerator = np.exp(kappa * (np.cos(x - mu) + 1)) - 1
        denominator = np.exp(2 * kappa) - 1
        return numerator / denominator

def init_connect(con, par):
    k1 = 1.0 / (par.w1 ** 2)
    k2 = 1.0 / (par.w2 ** 2)
    for i in range(par.nbpts):
        x = 2 * np.pi * i / par.nbpts
        con[i] = (par.offcon +
                  par.a1 * vonmises(x, 0.0, k1) -
                  par.a2 * vonmises(x, 0.0, k2))
    # Calculate conwidth
    for i in range(1, par.nbpts - 1):
        if con[i - 1] + con[i + 1] > 2 * con[i]:
            par.conwidth = i
            break

def init_connect_delta(con, par):
    for i in range(par.nbpts):
        con[i] = -par.beta * 2 * np.pi / par.nbpts
    con[0] += par.alpha
    dx = 2 * np.pi / par.nbpts
    con[par.nbpts - 1] += par.D / (dx ** 2)
    con[1] += par.D / (dx ** 2)
    con[0] -= 2 * par.D / (dx ** 2)

def init_connect_cosine(con, par):
    for i in range(par.nbpts):
        x = 2 * np.pi * i / par.nbpts
        con[i] = par.J0 + par.J1 * np.cos(x)

def init_input(input_signal, par, pos):
    kin = 1.0 / (par.win ** 2)
    for i in range(par.nbpts):
        x = 2 * np.pi * i / par.nbpts
        input_signal[i] = par.offin + par.ain * vonmises(x, pos, kin)

def deriv_fft(array, input_signal, par, fft_struct):
    # Perform convolution using FFT
    fft_array = scipy.fft.fft(array)
    convolved = scipy.fft.ifft(fft_array * fft_struct.fftcon).real
    convolved /= par.nbpts  # Normalize the inverse FFT result
    darray = convolved + 1 + input_signal
    # Thresholding and leakage term
    darray = np.maximum(darray, 0)
    darray -= array
    return darray

def dynamics_rk4step_fft(curstate, input_signal, par, fft_struct):
    dt = par.dt
    n = par.nbpts

    k1 = deriv_fft(curstate, input_signal, par, fft_struct)
    k2 = deriv_fft(curstate + dt * k1 / 2, input_signal, par, fft_struct)
    k3 = deriv_fft(curstate + dt * k2 / 2, input_signal, par, fft_struct)
    k4 = deriv_fft(curstate + dt * k3, input_signal, par, fft_struct)

    curstate += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def init_sw(curstate, input_signal, par, fft_struct):
    for step in range(int(20.0 / par.dt)):
        dynamics_rk4step_fft(curstate, input_signal, par, fft_struct)
        # Print curstate[0] every 100 steps
        if step % 100 == 0:
            print(f"Step {step}, curstate[0]: {curstate[0]}")
    
def bump_width(curstate, par):
    for i in range(1, par.nbpts - 1):
        if curstate[i] < 1e-6:
            return i
    return -1

def jump_vs_flow_input(curstate, input_signal, par, fft_struct, model, jump_dist):
    dim = par.nbpts

    # Initialize state
    curstate[:] = 0
    input_signal[:] = 0
    curstate[0] = 10.0
    curstate[1] = 10.0
    curstate[-1] = 10.0

    init_sw(curstate, input_signal, par, fft_struct)

    if curstate[0] > 1e6:
        print("curstate[0] too large, exiting.")
        return

    max_val = np.max(curstate)
    min_val = np.min(curstate)
    if abs(max_val - min_val) < 1e-4:
        print("Max and min values too close, exiting.")
        return

    bw = 2 * np.pi * bump_width(curstate, par) / par.nbpts
    if bw > np.pi / 4 or bw < 3 * np.pi / 16:
        print("Bump width out of acceptable range, exiting.")
        return

    curstate_ini = curstate.copy()

    nb_width = 160
    nb_amp = 40
    nb_amp_fine = 10
    sta_width = 0.01
    sto_width = 3 * np.pi / 4
    sta_amp = 0.01
    sto_amp = 5.0 * 3
    if model == 'delta':
        sto_amp = 0.4 * 3

    for i in range(nb_width):
        width = i * (sto_width - sta_width) / nb_width + sta_width
        kin = 1.0 / (width ** 2)
        out = 0
        for j in range(nb_amp):
            amp = j * (sto_amp - sta_amp) / nb_amp + sta_amp
            # Prepare input
            for k in range(dim):
                x = 2 * np.pi * k / dim
                input_signal[k] = amp * vonmises(x, 0.0, kin)
                curstate[k] = curstate_ini[k]
            init_sw(curstate, input_signal, par, fft_struct)
            ampli_ini = curstate[0]

            # Apply input at new position
            for k in range(dim):
                x = 2 * np.pi * k / dim
                input_signal[k] = amp * vonmises(x, np.deg2rad(jump_dist), kin)
                curstate[k] = curstate_ini[k]

            check = 0
            for _ in range(int(20.0 / par.dt)):
                dynamics_rk4step_fft(curstate, input_signal, par, fft_struct)
                max_pos = np.argmax(curstate)
                if dim // 16 < max_pos < 7 * dim // 8:
                    check = 1
                    break
            if check != out:
                for k in range(nb_amp_fine):
                    amp_fine = (j - 1 + k / nb_amp_fine) * (sto_amp - sta_amp) / nb_amp + sta_amp
                    # Prepare input for fine amplitude
                    for l in range(dim):
                        x = 2 * np.pi * l / dim
                        input_signal[l] = amp_fine * vonmises(x, 0.0, kin)
                        curstate[l] = curstate_ini[l]
                    init_sw(curstate, input_signal, par, fft_struct)
                    ampli_ini = curstate[0]

                    # Apply input at new position
                    for l in range(dim):
                        x = 2 * np.pi * l / dim
                        input_signal[l] = amp_fine * vonmises(x, np.deg2rad(jump_dist), kin)
                        curstate[l] = curstate_ini[l]

                    for _ in range(int(20.0 / par.dt)):
                        dynamics_rk4step_fft(curstate, input_signal, par, fft_struct)
                        max_pos = np.argmax(curstate)
                        if dim // 16 < max_pos < 7 * dim // 8:
                            break
            out = check
            print(f"{ampli_ini} {width} {amp} {check}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python simulation.py [delta|cosine] jump_distance")
        return

    model = sys.argv[1]
    jump_dist = float(sys.argv[2])  # in degrees

    par = Parasw()
    fft_struct = FFT(par.nbpts)

    # Initialize connectivity
    con = fft_struct.connect
    if model == 'delta':
        par.alpha = 3.0
        par.beta = 20.0
        par.D = 0.10
        init_connect_delta(con, par)
    elif model == 'cosine':
        par.J0 = -0.2
        par.J1 = 0.15
        init_connect_cosine(con, par)
    else:
        print("Model not recognized")
        return

    # Compute FFT of the connectivity
    fft_struct.fftcon = scipy.fft.fft(con)
    plt.figure()
    plt.plot(con)
    plt.title('Connectivity Profile')
    plt.xlabel('Neuron Index')
    plt.ylabel('Connectivity Strength')
    plt.show()

    # Initialize state and input
    curstate = np.zeros(par.nbpts)
    input_signal = np.zeros(par.nbpts)

    # Run the simulation
    jump_vs_flow_input(curstate, input_signal, par, fft_struct, model, jump_dist)

    # Optional: Plot the final state
    plt.figure()
    plt.plot(curstate)
    plt.xlabel('Neuron Index')
    plt.ylabel('Activity')
    plt.title('Final State after Simulation')
    plt.show()

if __name__ == "__main__":
    main()