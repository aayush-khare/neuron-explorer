from math import exp
import numpy as np

class HvcInterNeuron:
    """
    Implementation of the HVC(I) neurons using a 1 compartment
    model with temperature dependence.

    """

    CAPACITANCE = 1.0 # muF/cm^2
    TEMPERATURE_BASELINE = 40.0
    #RK4 constants

    B1 = 1.0 / 6.0
    B2 = 1.0 / 3.0
    B3 = 1.0 / 3.0
    B4 = 1.0 / 6.0

    B1_ = 1.0 / 8.0
    B2_ = 3.0 / 8.0
    B3_ = 3.0 / 8.0
    B4_ = 1.0 / 8.0

    def __init__(self, temperature_expt, q_gate, q_cond):
        
        # temperature related parameters coming from the control bars

        self.temperature_expt = temperature_expt
        self.q_gate = q_gate
        self.q_cond = q_cond
        
        self.q_fac_1 = self.q_cond ** (0.1 * (self.TEMPERATURE_BASELINE - self.temperature_expt))
        self.q_fac_2 = self.q_gate ** (0.1 * (self.TEMPERATURE_BASELINE - self.temperature_expt))
        self.rev_fac = ((273.15 + self.temperature_expt) / (273.15 + self.TEMPERATURE_BASELINE))
               
        self.g_na = 100.0 / self.q_fac_1
        self.g_kdr = 20.0 / self.q_fac_1
        self.g_kht = 500.0 / self.q_fac_1
        self.g_leak = 0.10 / self.q_fac_1
        
        self.e_na = 55.0 * self.rev_fac
        self.e_k = -80.0 * self.rev_fac
        self.e_leak = -65.0 * self.rev_fac
        self.e_inhib = -75.0 * self.rev_fac

        self.t_array = None
        self.current_stimulus_array = None
        self.excitatory_synapse_stimulus_array = None
        self.inhibitory_synapse_stimulus_array = None
        self.noise_freq= None
        self.noise_strength= None

        self.state_history = None

    def get_initial_state(self):

        v = -65.0
        n = 0.125
        m = 0.0
        h = 0.99
        w = 0.0
        ge = 0.0
        gi = 0.0
        noise_event_time = 0.0
        ge_noise = 0.0
        gi_noise = 0.0
        i_inj = 0.0

        return [v, n, m, h, w, ge, gi, noise_event_time, ge_noise, gi_noise, i_inj]

    def nmh_update(self, gate, alpha_gate, beta_gate):
        return alpha_gate - gate * (alpha_gate + beta_gate)
    
    def w_update(self, gate, gate_inf, tau_gate):
        return (gate_inf - gate) / tau_gate

    def alpha_n(self, v):
        return 0.15 * (15.0 + v) / (1.0 - exp(-(15.0 + v) / 10.0)) / self.q_fac_2

    def beta_n(self, v):
        return 0.2 * exp(-(v + 25.0) / 80.0) / self.q_fac_2
    
    def alpha_m(self, v):
        return (v + 22.0) / (1.0 - exp(-(v + 22.0) / 10.0)) / self.q_fac_2

    def beta_m(self, v):
        return 40.0 * exp(-(v + 47.0) / 18.0) / self.q_fac_2
    
    def alpha_h(self, v):
        return 0.7 * exp(-(v + 34.0) / 20.0) / self.q_fac_2
    
    def beta_h(self, v):
        return 10.0 / (1.0 + exp(-(v + 4.0) / 10.0)) / self.q_fac_2
    
    def w_inf(self, v):
        return 1.0 / (1.0 + exp(-v / 5.0))
    
    def tau_w(self):
        return self.q_fac_2
    
    def dv_dt(self, current, v, n, m, h, w, ge, gi, gen, gin):

        i_injected = current / (self.CAPACITANCE)
        
        i_leak = self.g_leak * (self.e_leak - v)
        i_na = self.g_na * m**3 * h *(self.e_na - v)
        i_k = self.g_kdr * n**4 * (self.e_k - v)
        i_kht = self.g_kht * w * (self.e_k - v)

        i_ion_total = (i_leak + i_na + i_k + i_kht) / self.CAPACITANCE

        i_excit = -ge * v
        i_inhib = gi * (self.e_inhib - v)
        i_excit_noise = -gen * v
        i_inhib_noise = gin * (self.e_inhib - v)

        i_synapse_total = (i_excit + i_inhib + i_excit_noise + i_inhib_noise) / (self.CAPACITANCE * self.q_fac_1)
        
        return i_injected + i_ion_total + i_synapse_total
    
    def g_update_for_num_method(self, g, tau_g, step_size):
        return g * exp(-step_size / (tau_g * self.q_fac_2))

    def generate_poisson_event(self, event_time, freq):
        rand_num = np.random.uniform(0.0, 1.0)
        event_time = event_time - (np.log(rand_num) / freq)

        return event_time
    
    def g_noise_update(self, current_time, event_time, e_cond, i_cond, g_max, freq, step_size):
        np.random.seed(int(current_time / step_size) + 201402)
        event_count = 0

        while True:
            if current_time + (0.5 * step_size) <= 1000.0 * event_time < current_time + (1.5 * step_size):
                event_count += 1
                event_time = self.generate_poisson_event(event_time, freq / self.q_fac_2)
            elif 1000.0 * event_time < current_time + (0.5 * step_size):
                event_time = self.generate_poisson_event(event_time, freq / self.q_fac_2)
            else:
                break
        
        kick_strength = 0
        p = 0
        num_exc = 0
        num_inh = 0

        if event_count == 0:
            e_cond = e_cond * exp(-step_size / (2 * self.q_fac_2))
            i_cond = i_cond * exp(-step_size / (2 * self.q_fac_2))

        else:
            for _ in range(event_count):
                p = np.random.uniform(0, 1)
                if p < 0.5:
                    num_exc += 1
                else:
                    num_inh += 1
            
            if num_exc != 0:
                for _ in range(num_exc):
                    kick_strength += np.random.uniform(0.0, 1.0)*g_max
                e_cond = e_cond + kick_strength

                kick_strength = 0
                num_exc = 0
            
            else:
                e_cond = e_cond * exp(-step_size / (2 * self.q_fac_2))

            if num_inh != 0:
                for _ in range(num_inh):
                    kick_strength += np.random.uniform(0.0, 1.0)*g_max
                i_cond = i_cond + kick_strength

                kick_strength = 0
                num_inh = 0
            
            else:
                i_cond = i_cond * exp(-step_size / (2 * self.q_fac_2))

            event_count = 0
        
        return event_time, e_cond, i_cond

    def g_ext_update(self, current_time, event_time, cond, g_max, freq, step_size):

        np.random.seed(int(current_time / step_size) + 101)
        event_count = 0

        while True:
            if current_time + (0.5 * step_size) <= 1000.0 * event_time < current_time + (1.5 * step_size):
                event_count += 1
                event_time = self.generate_poisson_event(event_time, freq)
            elif 1000.0 * event_time < current_time + (0.5 * step_size):
                event_time = self.generate_poisson_event(event_time, freq)
            else:
                break

        kick_strength = 0

        if event_count == 0:
            cond = cond * exp(-step_size / (2 * self.q_fac_2))
        else:
            for _ in range(event_count):
                kick_strength += np.random.uniform(0.0, 1.0) * g_max

            cond = cond + kick_strength
            event_count = 0

        return event_time, cond
    
    def rk4_step_alt(self, step_size, state):

        h1v = state[0]
        h1n = state[1]
        h1m = state[2]
        h1h = state[3]
        h1w = state[4]

        h1_ge = state[5]
        h1_gi = state[6]
        t_noise = state[7]
        h1_gen = state[8]
        h1_gin = state[9]
        h1_i = state[10]

        a1n = self.nmh_update(h1n, self.alpha_n(h1v), self.beta_n(h1v))
        a1m = self.nmh_update(h1m, self.alpha_m(h1v), self.beta_m(h1v))
        a1h = self.nmh_update(h1h, self.alpha_h(h1v), self.beta_h(h1v))
        a1w = self.w_update(h1w, self.w_inf(h1v), self.tau_w())
        a1v = self.dv_dt(h1_i, h1v, h1n, h1m, h1h, h1w, h1_ge, h1_gi, h1_gen, h1_gin)

        h2n = h1n + step_size * a1n / 3.0
        h2m = h1m + step_size * a1m / 3.0
        h2h = h1h + step_size * a1h / 3.0
        h2w = h1w + step_size * a1w / 3.0
        h2v = h1v + step_size * a1v / 3.0

        h2_ge = self.g_update_for_num_method(h1_ge, 2, step_size / 3)
        h2_gi = self.g_update_for_num_method(h1_gi, 2, step_size / 3)
        h2_gen = self.g_update_for_num_method(h1_gen, 2, step_size / 3)
        h2_gin = self.g_update_for_num_method(h1_gin, 2, step_size / 3)
        h2_i = state[10]

        a2n = self.nmh_update(h2n, self.alpha_n(h2v), self.beta_n(h2v))
        a2m = self.nmh_update(h2m, self.alpha_m(h2v), self.beta_m(h2v))
        a2h = self.nmh_update(h2h, self.alpha_h(h2v), self.beta_h(h2v))
        a2w = self.w_update(h2w, self.w_inf(h2v), self.tau_w())
        a2v = self.dv_dt(h2_i, h2v, h2n, h2m, h2h, h2w, h2_ge, h2_gi, h2_gen, h2_gin)

        h3n = h1n + step_size * (-a1n / 3.0 + a2n)
        h3m = h1m + step_size * (-a1m / 3.0 + a2m)
        h3h = h1h + step_size * (-a1h / 3.0 + a2h)
        h3w = h1w + step_size * (-a1w / 3.0 + a2w)
        h3v = h1v + step_size * (-a1v / 3.0 + a2v)

        h3_ge = self.g_update_for_num_method(h1_ge, 2, 2 * step_size / 3)
        h3_gi = self.g_update_for_num_method(h1_gi, 2, 2 * step_size / 3)
        h3_gen = self.g_update_for_num_method(h1_gen, 2, 2 * step_size / 3)
        h3_gin = self.g_update_for_num_method(h1_gin, 2, 2 * step_size / 3)
        h3_i = state[10]

        a3n = self.nmh_update(h3n, self.alpha_n(h3v), self.beta_n(h3v))
        a3m = self.nmh_update(h3m, self.alpha_m(h3v), self.beta_m(h3v))
        a3h = self.nmh_update(h3h, self.alpha_h(h3v), self.beta_h(h3v))
        a3w = self.w_update(h3w, self.w_inf(h3v), self.tau_w())
        a3v = self.dv_dt(h3_i, h3v, h3n, h3m, h3h, h3w, h3_ge, h3_gi, h3_gen, h3_gin)

        h4n = h1n + step_size * (a1n - a2n + a3n)
        h4m = h1m + step_size * (a1m - a2m + a3m)
        h4h = h1h + step_size * (a1h - a2h + a3h)
        h4w = h1w + step_size * (a1w - a2w + a3w)
        h4v = h1v + step_size * (a1v - a2v + a3v)

        h4_ge = self.g_update_for_num_method(h1_ge, 2, step_size)
        h4_gi = self.g_update_for_num_method(h1_gi, 2, step_size)
        h4_gen = self.g_update_for_num_method(h1_gen, 2, step_size)
        h4_gin = self.g_update_for_num_method(h1_gin, 2, step_size)
        h4_i = state[10]

        a4n = self.nmh_update(h4n, self.alpha_n(h4v), self.beta_n(h4v))
        a4m = self.nmh_update(h4m, self.alpha_m(h4v), self.beta_m(h4v))
        a4h = self.nmh_update(h4h, self.alpha_h(h4v), self.beta_h(h4v))
        a4w = self.w_update(h4w, self.w_inf(h4v), self.tau_w())
        a4v = self.dv_dt(h4_i, h4v, h4n, h4m, h4h, h4w, h4_ge, h4_gi, h4_gen, h4_gin)

        h5n = h1n + step_size * (self.B1_ * a1n + self.B2_ * a2n + self.B3_ * a3n + self.B4_ * a4n)
        h5m = h1m + step_size * (self.B1_ * a1m + self.B2_ * a2m + self.B3_ * a3m + self.B4_ * a4m)
        h5h = h1h + step_size * (self.B1_ * a1h + self.B2_ * a2h + self.B3_ * a3h + self.B4_ * a4h)
        h5w = h1w + step_size * (self.B1_ * a1w + self.B2_ * a2w + self.B3_ * a3w + self.B4_ * a4w)
        h5v = h1v + step_size * (self.B1_ * a1v + self.B2_ * a2v + self.B3_ * a3v + self.B4_ * a4v)

        return [h5v, h5n, h5m, h5h, h5w, h1_ge, h1_gi, t_noise, h1_gen, h1_gin, h1_i]
    
    def rk4_step(self, step_size, state):

        h1v = state[0]
        h1n = state[1]
        h1m = state[2]
        h1h = state[3]
        h1w = state[4]

        h1_ge = state[5]
        h1_gi = state[6]
        t_noise = state[7]
        h1_gen = state[8]
        h1_gin = state[9]
        h1_i = state[10]

        a1n = self.nmh_update(h1n, self.alpha_n(h1v), self.beta_n(h1v))
        a1m = self.nmh_update(h1m, self.alpha_m(h1v), self.beta_m(h1v))
        a1h = self.nmh_update(h1h, self.alpha_h(h1v), self.beta_h(h1v))
        a1w = self.w_update(h1w, self.w_inf(h1v), self.tau_w())
        a1v = self.dv_dt(h1_i, h1v, h1n, h1m, h1h, h1w, h1_ge, h1_gi, h1_gen, h1_gin)

        h2n = h1n + step_size * a1n / 2.0
        h2m = h1m + step_size * a1m / 2.0
        h2h = h1h + step_size * a1h / 2.0
        h2w = h1w + step_size * a1w / 2.0
        h2v = h1v + step_size * a1v / 2.0

        h2_ge = self.g_update_for_num_method(h1_ge, 2, 0.5 * step_size)
        h2_gi = self.g_update_for_num_method(h1_gi, 2, 0.5 * step_size)
        h2_gen = self.g_update_for_num_method(h1_gen, 2, 0.5 * step_size)
        h2_gin = self.g_update_for_num_method(h1_gin, 2, 0.5 * step_size)
        h2_i = state[10]

        a2n = self.nmh_update(h2n, self.alpha_n(h2v), self.beta_n(h2v))
        a2m = self.nmh_update(h2m, self.alpha_m(h2v), self.beta_m(h2v))
        a2h = self.nmh_update(h2h, self.alpha_h(h2v), self.beta_h(h2v))
        a2w = self.w_update(h2w, self.w_inf(h2v), self.tau_w())
        a2v = self.dv_dt(h2_i, h2v, h2n, h2m, h2h, h2w, h2_ge, h2_gi, h2_gen, h2_gin)

        h3n = h1n + step_size * a2n / 2.0
        h3m = h1m + step_size * a2m / 2.0
        h3h = h1h + step_size * a2h / 2.0
        h3w = h1w + step_size * a2w / 2.0
        h3v = h1v + step_size * a2v / 2.0

        h3_ge = self.g_update_for_num_method(h1_ge, 2, 0.5 * step_size)
        h3_gi = self.g_update_for_num_method(h1_gi, 2, 0.5 * step_size)
        h3_gen = self.g_update_for_num_method(h1_gen, 2, 0.5 * step_size)
        h3_gin = self.g_update_for_num_method(h1_gin, 2, 0.5 * step_size)
        h3_i = state[10]

        a3n = self.nmh_update(h3n, self.alpha_n(h3v), self.beta_n(h3v))
        a3m = self.nmh_update(h3m, self.alpha_m(h3v), self.beta_m(h3v))
        a3h = self.nmh_update(h3h, self.alpha_h(h3v), self.beta_h(h3v))
        a3w = self.w_update(h3w, self.w_inf(h3v), self.tau_w())
        a3v = self.dv_dt(h3_i, h3v, h3n, h3m, h3h, h3w, h3_ge, h3_gi, h3_gen, h3_gin)

        h4n = h1n + step_size * a3n
        h4m = h1m + step_size * a3m
        h4h = h1h + step_size * a3h
        h4w = h1w + step_size * a3w
        h4v = h1v + step_size * a3v

        h4_ge = self.g_update_for_num_method(h1_ge, 2, step_size)
        h4_gi = self.g_update_for_num_method(h1_gi, 2, step_size)
        h4_gen = self.g_update_for_num_method(h1_gen, 2, step_size)
        h4_gin = self.g_update_for_num_method(h1_gin, 2, step_size)
        h4_i = state[10]

        a4n = self.nmh_update(h4n, self.alpha_n(h4v), self.beta_n(h4v))
        a4m = self.nmh_update(h4m, self.alpha_m(h4v), self.beta_m(h4v))
        a4h = self.nmh_update(h4h, self.alpha_h(h4v), self.beta_h(h4v))
        a4w = self.w_update(h4w, self.w_inf(h4v), self.tau_w())
        a4v = self.dv_dt(h4_i, h4v, h4n, h4m, h4h, h4w, h4_ge, h4_gi, h4_gen, h4_gin)

        h5n = h1n + step_size * (self.B1 * a1n + self.B2 * a2n + self.B3 * a3n + self.B4 * a4n)
        h5m = h1m + step_size * (self.B1 * a1m + self.B2 * a2m + self.B3 * a3m + self.B4 * a4m)
        h5h = h1h + step_size * (self.B1 * a1h + self.B2 * a2h + self.B3 * a3h + self.B4 * a4h)
        h5w = h1w + step_size * (self.B1 * a1w + self.B2 * a2w + self.B3 * a3w + self.B4 * a4w)
        h5v = h1v + step_size * (self.B1 * a1v + self.B2 * a2v + self.B3 * a3v + self.B4 * a4v)

        return [h5v, h5n, h5m, h5h, h5w, h1_ge, h1_gi, t_noise, h1_gen, h1_gin, h1_i]
        
    def simulate(self,
                 t_array,
                 step_size,
                 current_stimulus_array=None,
                 excitatory_synapse_stimulus_array=None,
                 inhibitory_synapse_stimulus_array=None,
                 noise_freq=None,
                 noise_strength=None,
                 initial_state=None):
        
        if initial_state is None:
            initial_state = self.get_initial_state()

        self.t_array = t_array
        self.current_stimulus_array = current_stimulus_array
        self.excitatory_synapse_stimulus_array = excitatory_synapse_stimulus_array
        self.inhibitory_synapse_stimulus_array = inhibitory_synapse_stimulus_array
        self.noise_freq= noise_freq
        self.noise_strength= noise_strength

        num_steps = len(t_array)
        state_size = len(initial_state)
        state_history = np.zeros((num_steps, state_size))

        current_state = initial_state.copy()
        state_history[0] = current_state

        for i in range(1, num_steps):
            t = t_array[i]
            if current_stimulus_array is not None:
                current_state[10] = current_stimulus_array[i]

            if excitatory_synapse_stimulus_array is not None:
                current_state[5] = excitatory_synapse_stimulus_array[i]
                current_state[6] = inhibitory_synapse_stimulus_array[i]

            if noise_freq is not None:
                current_state[7], current_state[8], current_state[9] = self.g_noise_update(t,
                                                                                           current_state[7],
                                                                                           current_state[8],
                                                                                           current_state[9],
                                                                                           noise_strength,
                                                                                           noise_freq,
                                                                                           step_size)
                
            current_state = self.rk4_step_alt(step_size, current_state)

            state_history[i] = current_state

        self.state_history = state_history
        return state_history
    