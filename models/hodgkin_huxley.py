from math import exp
import numpy as np

class HodgkinHuxley:
    """
    Implementation of the Hodgkin-Huxley neuron model
    with temperature dependence.

    """

    CAPACITANCE = 1.0 # muF/cm^2
    TEMPERATURE_BASELINE = 6.3

    def __init__(self, temperature_expt, q_gate, q_cond):
        
        # temperature related parameters coming from the control bars

        self.temperature_expt = temperature_expt
        self.q_gate = q_gate
        self.q_cond = q_cond

        self.q_fac_1 = self.q_cond ** (0.1 * (self.TEMPERATURE_BASELINE - self.temperature_expt))
        self.q_fac_2 = self.q_gate ** (0.1 * (self.TEMPERATURE_BASELINE - self.temperature_expt))
        self.rev_fac = ((273.15 + self.temperature_expt) / (273.15 + self.TEMPERATURE_BASELINE))

        self.g_leak = 0.3 / self.q_fac_1
        self.g_na = 120.0 / self.q_fac_1
        self.g_k = 36.0 / self.q_fac_1

        self.e_leak = -65.0 * self.rev_fac
        self.e_na = 50.0 * self.rev_fac
        self.e_k = -77.0 * self.rev_fac

        self.t_array = None
        self.current_stimulus_array = None

        self.state_history = None

    def get_initial_state(self):
        v = -65.0
        n = 0.3
        m = 0.01
        h = 0.5
        i_inj = 0.0

        return [v, n, m, h, i_inj]
  
    def alpha_n(self, v):

        return (0.01 * (v + 55.0) / (1 - np.exp(-(v + 55.0) / 10.0))) / self.q_fac_2
        
    def beta_n(self, v):

        return 0.125 * np.exp(-(v + 65.0) / 80.0) / self.q_fac_2
    
    def alpha_h(self, v):

        return 0.07 * np.exp(-(v + 65.0) / 20.0) / self.q_fac_2

    def beta_h(self, v):

        return (1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))) / self.q_fac_2
    
    def alpha_m(self, v):
        
        return (0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))) / self.q_fac_2
    
    def beta_m(self, v):

        return 4.0 * np.exp(-(v + 65.0) / 18.0) / self.q_fac_2
    
    def nmh_update(self, gate, alpha, beta):

        return alpha * (1 - gate) - beta * gate

    def dv_dt(self, v, n, m, h, current):

        i_leak = self.g_leak * (self.e_leak - v)
        i_na = self.g_na * m * m * m * h * (self.e_na - v)
        i_k = self.g_k * n * n * n * n * (self.e_k - v)

        i_ion_total = (i_leak + i_na + i_k) / self.CAPACITANCE
        i_injected = current / self.CAPACITANCE

        return i_ion_total + i_injected
    
    def rk4_step(self, step_size, state):

        h1v = state[0]
        h1n = state[1]
        h1m = state[2]
        h1h = state[3]
        h1i = state[4]

        a1n = self.nmh_update(h1n, self.alpha_n(h1v), self.beta_n(h1v))
        a1m = self.nmh_update(h1m, self.alpha_m(h1v), self.beta_m(h1v))
        a1h = self.nmh_update(h1h, self.alpha_h(h1v), self.beta_h(h1v))
        a1v = self.dv_dt(h1v, h1n, h1m, h1h, h1i)

        h2n = h1n + step_size * a1n / 2.0
        h2m = h1m + step_size * a1m / 2.0
        h2h = h1h + step_size * a1h / 2.0
        h2v = h1v + step_size * a1v / 2.0
        h2i = state[4]

        a2n = self.nmh_update(h2n, self.alpha_n(h2v), self.beta_n(h2v))
        a2m = self.nmh_update(h2m, self.alpha_m(h2v), self.beta_m(h2v))
        a2h = self.nmh_update(h2h, self.alpha_h(h2v), self.beta_h(h2v))
        a2v = self.dv_dt(h2v, h2n, h2m, h2h, h2i)

        h3n = h1n + step_size * a2n / 2.0
        h3m = h1m + step_size * a2m / 2.0
        h3h = h1h + step_size * a2h / 2.0
        h3v = h1v + step_size * a2v / 2.0
        h3i = state[4]

        a3n = self.nmh_update(h3n, self.alpha_n(h3v), self.beta_n(h3v))
        a3m = self.nmh_update(h3m, self.alpha_m(h3v), self.beta_m(h3v))
        a3h = self.nmh_update(h3h, self.alpha_h(h3v), self.beta_h(h3v))
        a3v = self.dv_dt(h3v, h3n, h3m, h3h, h3i)

        h4n = h1n + step_size * a3n
        h4m = h1m + step_size * a3m
        h4h = h1h + step_size * a3h
        h4v = h1v + step_size * a3v
        h4i = state[4]

        a4n = self.nmh_update(h4n, self.alpha_n(h4v), self.beta_n(h4v))
        a4m = self.nmh_update(h4m, self.alpha_m(h4v), self.beta_m(h4v))
        a4h = self.nmh_update(h4h, self.alpha_h(h4v), self.beta_h(h4v))
        a4v = self.dv_dt(h4v, h4n, h4m, h4h, h4i)

        h5n = h1n + step_size * (a1n + 2 * a2n + 2 * a3n + a4n) / 6.0
        h5m = h1m + step_size * (a1m + 2 * a2m + 2 * a3m + a4m) / 6.0
        h5h = h1h + step_size * (a1h + 2 * a2h + 2 * a3h + a4h) / 6.0
        h5v = h1v + step_size * (a1v + 2 * a2v + 2 * a3v + a4v) / 6.0
        h5i = state[4]

        return [h5v, h5n, h5m, h5h, h5i]
    
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
                current_state[4] = current_stimulus_array[i]
            
            current_state = self.rk4_step(step_size, current_state)

            state_history[i] = current_state

        self.state_history = state_history
        return state_history