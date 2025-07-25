from math import exp, log
import numpy as np

class HvcProjectionNeuron:
    """
    Implementation of the HVC(RA) projecting neurons using a 2 compartment
    model (dendrite and soma) with temperature dependence.
    These neurons are of the excitatory type, sending inputs to other 
    HVC(RA) neurons, HVC Interneurons (HVC(I)) and to neurons in the 
    downstream brain region RA (reason why they are named so)

    """

    CAPACITANCE = 1.0 # muF/cm^2
    AREA_SOMA = 5000.0 
    AREA_DEND = 10000.0
    TEMPERATURE_BASELINE = 40.0 

    #AN3D1 constants
    A1 = -0.4526683126055039
    A2 = -0.4842227708685013
    A3 = 1.9368910834740051

    B1 = 1.0 / 6.0
    B2 = -0.005430430675258792
    B3 = 2.0 / 3.0
    B4 = 0.1720970973419255

    def __init__(self, temperature_expt, q_gate, q_cond):
        
        # temperature related parameters coming from the control bars

        self.temperature_expt = temperature_expt
        self.q_gate = q_gate
        self.q_cond = q_cond
        
        self.q_fac_1 = self.q_cond ** (0.1 * (self.TEMPERATURE_BASELINE - self.temperature_expt))
        self.q_fac_2 = self.q_gate ** (0.1 * (self.TEMPERATURE_BASELINE - self.temperature_expt))
        self.rev_fac = ((273.15 + self.temperature_expt) / (273.15 + self.TEMPERATURE_BASELINE))
       
        self.coupling = 130.0 * self.q_fac_1 # coupling resistance between compartments MOhm
        
        self.g_ca = 55.0 / self.q_fac_1
        self.g_cak = 150.0 / self.q_fac_1
        self.g_cabk = 100.0 / self.q_fac_1
        self.g_na = 60.0 / self.q_fac_1
        self.g_k = 8.0 / self.q_fac_1
        self.g_leak = 0.10 / self.q_fac_1
        
        self.e_na = 55.0 * self.rev_fac
        self.e_k = -90.0 * self.rev_fac
        self.e_ca = 120.0 * self.rev_fac
        self.e_leak = -80.0 * self.rev_fac
        self.e_inhib = -80.0 * self.rev_fac

        self.tau_r = 1.0 * self.q_fac_2
        self.tau_c = 15.0 * self.q_fac_2
        self.tau_bk = 2.0 * self.q_fac_2

        self.t_array = None
        self.current_stimulus_array = None
        self.excitatory_synapse_stimulus_array = None
        self.inhibitory_synapse_stimulus_array = None
        self.external_input_freq= None
        self.external_input_strength= None
        self.noise_freq= None
        self.noise_strength= None

        self.state_history = None

    def get_initial_state(self):

        vd = -80.0
        vs = -80.0
        n = 0.01101284
        h = 0.9932845
        r = 0.00055429
        c = 0.00000261762353
        c_bk = 0.00000261762353
        ca = 0.01689572
        v_half = max(-50.0, 72.0 - 30.0 * log(max(0.1, ca)))
        ge = 0.0
        gi = 0.0
        ext_event_time = 0.0
        gext = 0.0
        dend_noise_event_time = 0.0
        ge_noise_dend = 0.0
        ge_noise_soma = 0.0
        soma_noise_event_time = 0.0
        gi_noise_dend = 0.0
        gi_noise_soma = 0.0
        i_inj = 0.0

        return [vd,
                vs,
                n,
                h,
                r,
                c,
                c_bk, ca, v_half, ge, gi, ext_event_time, gext, dend_noise_event_time, ge_noise_dend, gi_noise_dend, soma_noise_event_time, ge_noise_soma, gi_noise_soma, i_inj]

    def nhrc_update(self, gate, alpha, tau_gate):
        return (alpha - gate) / tau_gate

    def ca_update(self, ca, vd, r):
        return 0.0002 * r**2 * self.g_ca * (self.e_ca - vd) - 0.02 * (ca / self.q_fac_1)

    def n_inf(self, vs):
        return 1.0 / (1.0 + exp(-(vs + 35.0) / 10.0))

    def h_inf(self, vs):
        return 1.0 / (1.0 + exp((vs + 45.0) / 7.0))

    def r_inf(self, vd):
        return 1.0 / (1.0 + exp(-(vd + 5.0) / 10.0))

    def c_inf(self, vd):
        return 1.0 / (1.0 + exp(-(vd - 10.0) / 7.0))

    def c_BK_inf(self, vd, v_half):
        return 1.0 / (1.0 + exp((v_half - vd) / 13.0))

    def tau_n(self, vs):
        return (0.1 + (0.5 / (1.0 + exp((vs + 27.0) / 15.0)))) * self.q_fac_2

    def tau_h(self, vs):
        return (0.1 + (0.75 / (1.0 + exp((40.5 + vs)/ 6.0)))) * self.q_fac_2
    
    def kvd(self, current, vd, vs, r, c, c_bk, ca, ge, gi, gext, gen, gin):

        i_coupling = 10**5 * (vs - vd) / (self.CAPACITANCE * self.AREA_DEND * self.coupling)

        i_injected = 10**5 * current / (self.CAPACITANCE * self.AREA_DEND)
        
        i_leak = self.g_leak * (self.e_leak - vd)
        i_ca = self.g_ca * r**2 *(self.e_ca - vd)
        i_cak = self.g_cak * (c / (1.0 + (6.0 / ca))) * (self.e_k - vd)
        i_cabk = self.g_cabk * c_bk * (self.e_k - vd)

        i_ion_total = (i_leak + i_ca + i_cak + i_cabk) / self.CAPACITANCE

        i_excit = -ge * vd
        i_inhib = gi * (self.e_inhib - vd)
        i_nif = -gext * vd
        i_excit_noise = -gen * vd
        i_inhib_noise = gin * (self.e_inhib - vd)

        i_synapse_total = (i_excit + i_inhib + i_nif + i_excit_noise + i_inhib_noise) / (self.CAPACITANCE * self.q_fac_1)
        
        return i_coupling + i_injected + i_ion_total + i_synapse_total

    def kvs(self, vs, vd, n, h, gen, gin):

        i_coupling = 10**5 * (vd - vs) / (self.CAPACITANCE * self.AREA_SOMA * self.coupling)
        
        i_leak = 0.5 * self.g_leak * (self.e_leak - vs)
        i_na = self.g_na * h * ((1/ (1 + exp(-(vs + 25.0) / 9.5)))**3 * (self.e_na - vs))
        i_k = self.g_k * n**4 * (self.e_k - vs)

        i_ion_total = (i_leak + i_na + i_k) / self.CAPACITANCE

        i_excit_noise = -gen * vs
        i_inhib_noise = gin * (self.e_inhib - vs)

        i_synapse_total = (i_excit_noise + i_inhib_noise) / (self.CAPACITANCE * self.q_fac_1)
        
        return i_coupling + i_ion_total + i_synapse_total
    
    def g_update_for_num_method(self, g, tau_g, step_size):
        return g * exp(-step_size / (tau_g * self.q_fac_2))

    def generate_poisson_event(self, event_time, freq):
        rand_num = np.random.uniform(0.0, 1.0)
        event_time = event_time - (np.log(rand_num) / freq)

        return event_time
    
    def g_noise_update(self, current_time, event_time, e_cond, i_cond, g_max, freq, step_size):
        np.random.seed(74391 + int(current_time / step_size))
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

    def an3d1_step(self, step_size, state):

        h1vd = state[0]
        h1vs = state[1]
        h1n = state[2]
        h1h = state[3]
        h1r = state[4]
        h1c = state[5]
        h1c_bk = state[6]
        h1ca = state[7]
        h1v_half = state[8]

        h1_ge = state[9]
        h1_gi = state[10]
        t_ext = state[11]
        h1_gext = state[12]
        t_n_d = state[13]
        h1_gen_d = state[14]
        h1_gin_d = state[15]
        t_n_s = state[16]
        h1_gen_s = state[17]
        h1_gin_s = state[18]
        h1_i = state[19]

        a1n = self.nhrc_update(h1n, self.n_inf(h1vs), self.tau_n(h1vs))
        a1h = self.nhrc_update(h1h, self.h_inf(h1vs), self.tau_h(h1vs))
        a1r = self.nhrc_update(h1r, self.r_inf(h1vd), self.tau_r)
        a1c = self.nhrc_update(h1c, self.c_inf(h1vd), self.tau_c)
        a1c_bk = self.nhrc_update(h1c_bk, self.c_BK_inf(h1vd, h1v_half), self.tau_bk)
        a1ca = self.ca_update(h1ca, h1vd, h1r)
        a1vd = self.kvd(h1_i, h1vd, h1vs, h1r, h1c, h1c_bk, h1ca, h1_ge, h1_gi, h1_gext, h1_gen_d, h1_gin_d)
        a1vs = self.kvs(h1vs, h1vd, h1n, h1h, h1_gen_s, h1_gin_s)

        h2n = h1n + step_size * a1n
        h2h = h1h + step_size * a1h
        h2r = h1r + step_size * a1r
        h2c = h1c + step_size * a1c
        h2c_bk = h1c_bk + step_size * a1c_bk
        h2ca = h1ca + step_size * a1ca
        h2vd = h1vd + step_size * a1vd
        h2vs = h1vs + step_size * a1vs
        h2v_half = max(-50.0, 72.0 - 30.0 * log(max(0.1, h2ca)))

        h2_ge = self.g_update_for_num_method(state[9], 2, step_size)
        h2_gi = self.g_update_for_num_method(state[10], 2, step_size)
        h2_gext = self.g_update_for_num_method(state[12], 2, step_size)
        h2_gen_d = self.g_update_for_num_method(state[14], 2, step_size)
        h2_gen_s = self.g_update_for_num_method(state[17], 2, step_size)
        h2_gin_d = self.g_update_for_num_method(state[15], 2, step_size)
        h2_gin_s = self.g_update_for_num_method(state[18], 2, step_size)
        h2_i = state[19]

        a2n = self.nhrc_update(h2n, self.n_inf(h2vs), self.tau_n(h2vs))
        a2h = self.nhrc_update(h2h, self.h_inf(h2vs), self.tau_h(h2vs))
        a2r = self.nhrc_update(h2r, self.r_inf(h2vd), self.tau_r)
        a2c = self.nhrc_update(h2c, self.c_inf(h2vd), self.tau_c)
        a2c_bk = self.nhrc_update(h2c_bk, self.c_BK_inf(h2vd, h2v_half), self.tau_bk)
        a2ca = self.ca_update(h2ca, h2vd, h2r)
        a2vd = self.kvd(h2_i, h2vd, h2vs, h2r, h2c, h2c_bk, h2ca, h2_ge, h2_gi, h2_gext, h2_gen_d, h2_gin_d)
        a2vs = self.kvs(h2vs, h2vd, h2n, h2h, h2_gen_s, h2_gin_s)

        h3n = h1n + step_size * (3.0 * a1n + a2n) / 8.0
        h3h = h1h + step_size * (3.0 * a1h + a2h) / 8.0
        h3r = h1r + step_size * (3.0 * a1r + a2r) / 8.0
        h3c = h1c + step_size * (3.0 * a1c + a2c) / 8.0
        h3c_bk = h1c_bk + step_size * (3.0 * a1c_bk + a2c_bk) / 8.0
        h3ca = h1ca + step_size * (3.0 * a1ca + a2ca) / 8.0
        h3vd = h1vd + step_size * (3.0 * a1vd + a2vd) / 8.0
        h3vs = h1vs + step_size * (3.0 * a1vs + a2vs) / 8.0
        h3v_half = max(-50.0, 72.0 - 30.0 * log(max(0.1, h3ca)))

        h3_ge = self.g_update_for_num_method(state[9], 2, 0.5*step_size)
        h3_gi = self.g_update_for_num_method(state[10], 2, 0.5*step_size)
        h3_gext = self.g_update_for_num_method(state[12], 2, 0.5*step_size)
        h3_gen_d = self.g_update_for_num_method(state[14], 2, 0.5*step_size)
        h3_gen_s = self.g_update_for_num_method(state[17], 2, 0.5*step_size)
        h3_gin_d = self.g_update_for_num_method(state[15], 2, 0.5*step_size)
        h3_gin_s = self.g_update_for_num_method(state[18], 2, 0.5*step_size)
        h3_i = state[19]

        a3n = self.nhrc_update(h3n, self.n_inf(h3vs), self.tau_n(h3vs))
        a3h = self.nhrc_update(h3h, self.h_inf(h3vs), self.tau_h(h3vs))
        a3r = self.nhrc_update(h3r, self.r_inf(h3vd), self.tau_r)
        a3c = self.nhrc_update(h3c, self.c_inf(h3vd), self.tau_c)
        a3c_bk = self.nhrc_update(h3c_bk, self.c_BK_inf(h3vd, h3v_half), self.tau_bk)
        a3ca = self.ca_update(h3ca, h3vd, h3r)
        a3vd = self.kvd(h3_i, h3vd, h3vs, h3r, h3c, h3c_bk, h3ca, h3_ge, h3_gi, h3_gext, h3_gen_d, h3_gin_d)
        a3vs = self.kvs(h3vs, h3vd, h3n, h3h, h3_gen_s, h3_gin_s)

        h4n = h1n + step_size * (self.A1 * a1n + self.A2 * a2n + self.A3 * a3n)
        h4h = h1h + step_size * (self.A1 * a1h + self.A2 * a2h + self.A3 * a3h)
        h4r = h1r + step_size * (self.A1 * a1r + self.A2 * a2r + self.A3 * a3r)
        h4c = h1c + step_size * (self.A1 * a1c + self.A2 * a2c + self.A3 * a3c)
        h4c_bk = h1c_bk + step_size * (self.A1 * a1c_bk + self.A2 * a2c_bk + self.A3 * a3c_bk)
        h4ca = h1ca + step_size * (self.A1 * a1ca + self.A2 * a2ca + self.A3 * a3ca)
        h4vd = h1vd + step_size * (self.A1 * a1vd + self.A2 * a2vd + self.A3 * a3vd)
        h4vs = h1vs + step_size * (self.A1 * a1vs + self.A2 * a2vs + self.A3 * a3vs)
        h4v_half = max(-50.0, 72.0 - 30.0 * log(max(0.1, h4ca)))

        h4_ge = self.g_update_for_num_method(state[9], 2, step_size)
        h4_gi = self.g_update_for_num_method(state[10], 2, step_size)
        h4_gext = self.g_update_for_num_method(state[12], 2, step_size)
        h4_gen_d = self.g_update_for_num_method(state[14], 2, step_size)
        h4_gen_s = self.g_update_for_num_method(state[17], 2, step_size)
        h4_gin_d = self.g_update_for_num_method(state[15], 2, step_size)
        h4_gin_s = self.g_update_for_num_method(state[18], 2, step_size)
        h4_i = state[19]

        a4n = self.nhrc_update(h4n, self.n_inf(h4vs), self.tau_n(h4vs))
        a4h = self.nhrc_update(h4h, self.h_inf(h4vs), self.tau_h(h4vs))
        a4r = self.nhrc_update(h4r, self.r_inf(h4vd), self.tau_r)
        a4c = self.nhrc_update(h4c, self.c_inf(h4vd), self.tau_c)
        a4c_bk = self.nhrc_update(h4c_bk, self.c_BK_inf(h4vd, h4v_half), self.tau_bk)
        a4ca = self.ca_update(h4ca, h4vd, h4r)
        a4vd = self.kvd(h4_i, h4vd, h4vs, h4r, h4c, h4c_bk, h4ca, h4_ge, h4_gi, h4_gext, h4_gen_d, h4_gin_d)
        a4vs = self.kvs(h4vs, h4vd, h4n, h4h, h4_gen_s, h4_gin_s)

        h5n = h1n + step_size * (self.B1 * a1n + self.B2 * a2n + self.B3 * a3n + self.B4 * a4n)
        h5h = h1h + step_size * (self.B1 * a1h + self.B2 * a2h + self.B3 * a3h + self.B4 * a4h)
        h5r = h1r + step_size * (self.B1 * a1r + self.B2 * a2r + self.B3 * a3r + self.B4 * a4r)
        h5c = h1c + step_size * (self.B1 * a1c + self.B2 * a2c + self.B3 * a3c + self.B4 * a4c)
        h5c_bk = h1c_bk + step_size*(self.B1 * a1c_bk + self.B2 * a2c_bk + self.B3 * a3c_bk + self.B4 * a4c_bk)
        h5ca = h1ca + step_size * (self.B1 * a1ca + self.B2 * a2ca + self.B3 * a3ca + self.B4 * a4ca)
        h5vd = h1vd + step_size * (self.B1 * a1vd + self.B2 * a2vd + self.B3 * a3vd + self.B4 * a4vd)
        h5vs = h1vs + step_size * (self.B1 * a1vs + self.B2 * a2vs + self.B3 * a3vs + self.B4 * a4vs)

        return [h5vd, h5vs, h5n, h5h, h5r, h5c, h5c_bk, h5ca, h4v_half, h1_ge, h1_gi, t_ext, h1_gext, t_n_d, h1_gen_d, h1_gin_d, t_n_s, h1_gen_s, h1_gin_s, h1_i]
        
    def simulate(self,
                 t_array,
                 step_size,
                 current_stimulus_array=None,
                 excitatory_synapse_stimulus_array=None,
                 inhibitory_synapse_stimulus_array=None,
                 external_input_freq=None,
                 external_input_strength=None,
                 noise_freq=None,
                 noise_strength=None,
                 initial_state=None):
        
        if initial_state is None:
            initial_state = self.get_initial_state()

        self.t_array = t_array
        self.current_stimulus_array = current_stimulus_array
        self.excitatory_synapse_stimulus_array = excitatory_synapse_stimulus_array
        self.inhibitory_synapse_stimulus_array = inhibitory_synapse_stimulus_array
        self.external_input_freq= external_input_freq
        self.external_input_strength= external_input_strength
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
                current_state[19] = current_stimulus_array[i]

            if excitatory_synapse_stimulus_array is not None:
                current_state[9] = excitatory_synapse_stimulus_array[i]
                current_state[10] = inhibitory_synapse_stimulus_array[i]

            if noise_freq is not None:
                current_state[13], current_state[14], current_state[15] = self.g_noise_update(t,
                                                                                              current_state[13],
                                                                                              current_state[14],
                                                                                              current_state[15],
                                                                                              noise_strength,
                                                                                              noise_freq,
                                                                                              step_size)
                current_state[16], current_state[17], current_state[18] = self.g_noise_update(t,
                                                                                              current_state[16],
                                                                                              current_state[17],
                                                                                              current_state[18],
                                                                                              noise_strength,
                                                                                              noise_freq,
                                                                                              step_size)

            if external_input_freq is not None:
                current_state[11], current_state[12] = self.g_ext_update(t,
                                                                         current_state[11],
                                                                         current_state[12],
                                                                         external_input_strength,
                                                                         external_input_freq,
                                                                         step_size)

            current_state = self.an3d1_step(step_size, current_state)

            state_history[i] = current_state

        self.state_history = state_history
        return state_history
    