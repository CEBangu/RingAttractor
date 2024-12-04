import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import seaborn as sns
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SimulationParameters:
    nbpts: int          # Number of neurons
    dt: float           # Time step size
    w1: float           # Connectivity parameter
    w2: float           # Connectivity parameter
    win: float          # Width of external input
    a1: float           # Connectivity parameter
    a2: float           # Connectivity parameter
    beta: float         # Connectivity parameter
    alpha: float        # Connectivity parameter
    offcon: float       # Connectivity offset
    offin: float        # Input offset
    ain: float          # Input amplitude scaling
    D: float            # Diffusion parameter
    conwidth: int = 0   # Calculated connectivity width

class RingAttractor:
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.nbpts = params.nbpts
        self.theta = np.linspace(0, 2 * np.pi, self.nbpts, endpoint=False)
        self.connect = np.zeros(self.nbpts)
        self.input = np.zeros(self.nbpts)
        self.curstate = np.zeros(self.nbpts)
        self.buffer = np.zeros(8 * self.nbpts)  # Buffer for RK4 steps

    def vonmises(self, x: np.ndarray, mu: float, kappa: float) -> np.ndarray:
        """Compute the von Mises function."""
        if kappa > 5:
            return np.exp(kappa * (np.cos(x - mu) - 1))
        return (np.exp(kappa * (np.cos(x - mu) + 1)) - 1) / (np.exp(2 * kappa) - 1)

    def init_connect_delta(self):
        """Initialize connectivity with delta profile as in 'init_connect_delta'."""
        self.connect.fill(-self.params.beta * 2 * np.pi / self.nbpts)
        self.connect[0] += self.params.alpha
        dx = 2 * np.pi / self.nbpts
        self.connect[-1] += self.params.D / dx**2
        self.connect[1] += self.params.D / dx**2
        self.connect[0] -= 2 * self.params.D / dx**2

    def initialize_state(self):
        """Initialize the activity bump at three adjacent neurons."""
        self.curstate.fill(0)
        mu_init = 0.0  # Initial position in radians
        mu_index = int(mu_init / (2 * np.pi) * self.nbpts) % self.nbpts
        self.curstate[mu_index] = 10.0
        self.curstate[(mu_index - 1) % self.nbpts] = 10.0
        self.curstate[(mu_index + 1) % self.nbpts] = 10.0

    def deriv_fft(self, array: np.ndarray, input_signal: np.ndarray) -> np.ndarray:
        """
        Compute the derivative using FFT-based convolution.
        darray = (convolve(connect, array) / nbpts + 1 + input_signal) - array
        Apply thresholding to ensure non-negativity.
        """
        # Perform FFT-based convolution
        array_fft = np.fft.fft(array)
        connect_fft = np.fft.fft(self.connect)
        convolved = np.fft.ifft(array_fft * connect_fft).real
        net_input = convolved / self.nbpts + 1 + input_signal
        net_input = np.maximum(0, net_input)  # Thresholding
        darray = net_input - array
        return darray

    def dynamics_rk4step(self, array: np.ndarray, input_signal: np.ndarray) -> np.ndarray:
        """
        Perform one RK4 step to update the state.
        """
        k1 = self.deriv_fft(array, input_signal)
        k2 = self.deriv_fft(array + 0.5 * self.params.dt * k1, input_signal)
        k3 = self.deriv_fft(array + 0.5 * self.params.dt * k2, input_signal)
        k4 = self.deriv_fft(array + self.params.dt * k3, input_signal)
        return array + (self.params.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def apply_input(self, amplitude: float, inpos_deg: float, width: float):
        """
        Apply external input using a von Mises distribution.
        """
        inpos = np.deg2rad(inpos_deg)
        kin = 1.0 / width**2
        self.input = self.params.offin + self.params.ain * self.vonmises(self.theta, inpos, kin) * amplitude

    def simulate(self, steps: int, input_signal: np.ndarray):
        """
        Simulate the network for a given number of steps with the provided input_signal.
        """
        for _ in range(steps):
            self.curstate = self.dynamics_rk4step(self.curstate, input_signal)

def max_state(curstate: np.ndarray) -> Tuple[float, int]:
    """
    Identify the neuron with the maximum activity and its position.
    """
    max_val = np.max(curstate)
    max_pos = np.argmax(curstate)
    return max_val, max_pos

def angular_distance(theta1: float, theta2: float) -> float:
    """
    Calculate the minimum angular distance between two angles.
    """
    delta = np.abs(theta1 - theta2) % (2 * np.pi)
    return min(delta, 2 * np.pi - delta)

def categorize_regime(initial_peak_pos: float, final_peak_pos: float,
                     threshold_jump: float = np.pi / 4,
                     threshold_flow: float = np.pi / 16) -> str:
    """
    Categorize the regime based on the shift in peak position.
    
    Parameters:
    - initial_peak_pos: Angular position of the initial peak (radians)
    - final_peak_pos: Angular position of the final peak (radians)
    - threshold_jump: Angular threshold to classify as 'Jump' (default: 45 degrees)
    - threshold_flow: Angular threshold to classify as 'Flow' (default: 11.25 degrees)
    
    Returns:
    - 'Jump', 'Flow', or 'No Effect'
    """
    distance = angular_distance(initial_peak_pos, final_peak_pos)
    if distance > threshold_jump:
        return 'Jump'
    elif distance > threshold_flow:
        return 'Flow'
    else:
        return 'No Effect'

def run_simulation(params: SimulationParameters, inpos_deg: float = 120.0,
                   jump_dist_deg: float = 90.0) -> pd.DataFrame:
    """
    Run simulations across varying input amplitudes and categorize regimes.
    
    Parameters:
    - params: SimulationParameters object
    - inpos_deg: Degrees where external input is applied
    - jump_dist_deg: Degrees defining the jump distance
    
    Returns:
    - DataFrame containing amplitude, width, and detected regime
    """
    model = RingAttractor(params)
    model.init_connect_delta()
    model.initialize_state()
    
    # Identify initial peak position
    _, initial_peak_idx = max_state(model.curstate)
    initial_peak_pos = model.theta[initial_peak_idx]
    
    results = []
    
    # Define range of input amplitudes and widths
    input_amplitudes = [2e-4 * (2 ** a) for a in range(20)]  # 2e-4, 4e-4, ..., up to ~2.6
    input_amplitudes = [amp for amp in input_amplitudes if amp <= 0.8]  # Cap at 0.8
    input_widths = np.linspace(0.5, 3.0, 30)  # Widths from 0.5 to 3.0 (arbitrary units)
    
    for width in input_widths:
        for amp in input_amplitudes:
            # Reset the state
            model.initialize_state()
            
            # Apply external input
            model.apply_input(amplitude=amp, inpos_deg=inpos_deg, width=width)
            
            # Simulate for a fixed number of steps post-input
            simulation_steps = 1000  # Adjust based on desired simulation duration
            model.simulate(steps=simulation_steps, input_signal=model.input)
            
            # Identify final peak position
            _, final_peak_idx = max_state(model.curstate)
            final_peak_pos = model.theta[final_peak_idx]
            
            # Categorize regime
            regime = categorize_regime(initial_peak_pos, final_peak_pos)
            
            # Record the result
            results.append({
                'Amplitude': amp,
                'Width': width,
                'Regime': regime
            })
    
    df_results = pd.DataFrame(results)
    return df_results

# def generate_phase_diagram(df_results: pd.DataFrame):
#     """
#     Generate a phase diagram showing regimes across input amplitudes and widths.
    
#     Parameters:
#     - df_results: DataFrame containing simulation results
#     """
#     pivot_table = df_results.pivot('Width', 'Amplitude', 'Regime')
    
#     # Define a mapping from regimes to numerical values for coloring
#     regime_mapping = {'No Effect': 0, 'Flow': 1, 'Jump': 2}
#     pivot_numeric = pivot_table.replace(regime_mapping)
    
#     plt.figure(figsize=(14, 10))
#     sns.heatmap(pivot_numeric, cmap='viridis', cbar_kws={'ticks': [0, 1, 2], 'label': 'Regime'})
#     plt.xlabel('Input Amplitude')
#     plt.ylabel('Input Width')
#     plt.title('Phase Diagram: Regimes vs. Input Amplitude and Width')
    
#     # Create a custom legend
#     from matplotlib.patches import Patch
#     legend_elements = [Patch(facecolor='darkviolet', label='No Effect'),
#                        Patch(facecolor='blue', label='Flow'),
#                        Patch(facecolor='yellow', label='Jump')]
#     plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.tight_layout()
#     plt.show()

def main():
    # Define simulation parameters (adjust as per the C++ code's parameters)
    params = SimulationParameters(
        nbpts=256,        # Number of neurons
        dt=0.001,         # Time step size
        w1=1.0,           # Connectivity parameter
        w2=1.0,           # Connectivity parameter
        win=1.0,          # Width of external input
        a1=3.0,           # Connectivity parameter
        a2=20.0,          # Connectivity parameter
        beta=0.1,         # Connectivity parameter
        alpha=0.5,        # Connectivity parameter
        offcon=0.0,       # Connectivity offset
        offin=0.0,        # Input offset
        ain=1.0,          # Input amplitude scaling
        D=0.1             # Diffusion parameter
    )
    
    # Run simulations
    df_results = run_simulation(params, inpos_deg=0, jump_dist_deg=90.0)
    
    # Save results to CSV for further analysis if needed
    df_results.to_csv('simulation_results.csv', index=False)
    
    # Generate and display the phase diagram
    # generate_phase_diagram(df_results)
    
    # Optionally, display a summary of results
    print(df_results['Regime'].value_counts())

if __name__ == "__main__":
    main()