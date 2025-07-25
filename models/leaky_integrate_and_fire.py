from math import exp
import numpy as np

class LIF:
    """
    Implementation of the leaky integrate and fire model.

    """
    CAPACITANCE = 1.0 # muF/cm^2
    RESISTANCE = 10.0 #megaohm
    THRESHOLD = -50.0 # mV
    REST_STATE = -65.0
    E_LEAK = -65.0
    G_LEAK = 0.05

    def __init__(self):
        
        self.t_array = None
        self.current_stimulus_array = None

        self.state_history = None

    def get_initial_state(self):
        v = -65.0
        i_inj = 0.0

        return [v, i_inj]
    
    def dv_dt(self, v, current):
        i_leak = self.G_LEAK * (self.E_LEAK - v) / self.CAPACITANCE
        i_injected = current / self.CAPACITANCE

        return i_leak + i_injected
    
    def euler_step(self, step_size, state):
        
        h1v = state[0]
        h1i = state[1]

        if(h1v != 0.0):
            a1v = self.dv_dt(h1v, h1i)

            h2v = h1v + step_size * a1v
            h2i = state[1]
        
        elif(h1v == 0.0):
            h2v = self.REST_STATE
            h2i = state[1]

        if(h1v < self.THRESHOLD and h2v >= self.THRESHOLD):
            h2v = 0.0
            h2i = state[1]

        return [h2v, h2i]
    
    def simulate(self,
                 t_array,
                 step_size,
                 current_stimulus_array=None,
                 initial_state=None):
        
        if initial_state is None:
            initial_state = self.get_initial_state()

        self.t_array = t_array
        self.current_stimulus_array = current_stimulus_array

        num_steps = len(t_array)
        state_size = len(initial_state)
        state_history = np.zeros((num_steps, state_size))

        current_state = initial_state.copy()
        state_history[0] = current_state

        for i in range(1, num_steps):

            if current_stimulus_array is not None:
                current_state[1] = current_stimulus_array[i]
            
            current_state = self.euler_step(step_size, current_state)

            state_history[i] = current_state

        self.state_history = state_history
        return state_history