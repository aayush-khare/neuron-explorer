import numpy as np
import streamlit as st
from PIL import Image

from models.leaky_integrate_and_fire import LIF
from models.hodgkin_huxley import HodgkinHuxley
from models.hvc_projection_neuron import HvcProjectionNeuron
from models.hvc_interneuron import HvcInterNeuron

def create_current_stimulus_array(time_array, i_amp, i_start, i_end):
    '''
    Function for creating a pulse current input

    Returns:

    current stimulus : array
    numpy array containing current input values at different time points
    '''
    current_stimulus = np.zeros_like(time_array)
    start_id = np.argmin(np.abs(time_array - i_start))
    end_id = np.argmin(np.abs(time_array - i_end))
    current_stimulus[start_id:end_id] = i_amp
    return current_stimulus

def create_synapse_stimulus_array(time_array, temp, q_gate, g_max, g_start, step_size):
    '''
    Function for creating a single kick and decay type synaptic input

    Returns:

    synapse_stimulus : array
    numpy array containing values of the synaptic conductance strength at different time points
    '''
    synapse_stimulus = np.zeros_like(time_array)
    start_id = np.argmin(np.abs(time_array - g_start))
    q = pow(q_gate, 0.1 * (40.0 - temp))
    for i in range(len(time_array) - start_id):
        synapse_stimulus[start_id + i] = g_max * np.exp(-i * step_size / ( 2 * q ))

    return synapse_stimulus

def response_time(time, vd, vs, current_amplitude=None, current_input_start_time=None, excitatory_input_start_time=None, excitatory_input_strength=None, threshold=-20):
    """
    Assess whether the neuron spiked or not and return a rise time
    The rise time is defined as the time from input to reaching a peak somatic membrane 
    potential in case the peak is below threshold or the time from input to threshold membrane potential (-20 mV)

    Parameters:
    -----------

    Returns:
    --------

    """
    time = time.tolist()
    vs = vs.tolist()
    vd = vd.tolist()

    if current_amplitude is not None:
        if current_amplitude > 0:

            if max(vd) > threshold and max(vs) > threshold:
                rise_time = np.round(time[next(x[0] for x in enumerate(vs) if x[1] > threshold)] - current_input_start_time, 3)
                return rise_time
            elif max(vd) < threshold and max(vs) > threshold:
                rise_time = np.round(time[next(x[0] for x in enumerate(vs) if x[1] > threshold)] - current_input_start_time, 3)
                return rise_time
            elif max(vd) < threshold and max(vs) < threshold:
                rise_time = np.round(time[vs.index(max(vs))] - current_input_start_time, 3)
                return rise_time
    
    elif excitatory_input_strength is not None:
        if excitatory_input_strength > 0:
            if max(vd) > threshold and max(vs) > threshold:
                rise_time = np.round(time[next(x[0] for x in enumerate(vs) if x[1] > threshold)] - excitatory_input_start_time, 3)
                return rise_time
            elif max(vd) < threshold and max(vs) > threshold:
                rise_time = np.round(time[next(x[0] for x in enumerate(vs) if x[1] > threshold)] - excitatory_input_start_time, 3)
                return rise_time
            elif max(vd) < threshold and max(vs) < threshold:
                rise_time = np.round(time[vs.index(max(vs))] - excitatory_input_start_time, 3)
                return rise_time
            
def create_sidebar_controls_lif():
    '''
    Sidebar controls for the Leaky Integrate and Fire neuron model

    Returns:

    params : dict
    Dictionary containing all parameter values for the LIF model
    '''

    if 'lif_current_list' not in st.session_state:
        st.session_state.lif_current_list = []
    if 'lif_last_current' not in st.session_state:
        st.session_state.lif_last_current = 0.0
    
    if 'lif_frequency_list' not in st.session_state:
        st.session_state.lif_frequency_list = []
    if 'lif_reset_counter' not in st.session_state:
        st.session_state.lif_reset_counter = 0

    lif_reset_key = st.session_state.lif_reset_counter
    lif_reset_pressed = st.sidebar.button("Reset")

    if lif_reset_pressed:
        st.session_state.lif_current_list = []
        st.session_state.lif_frequency_list = []
        st.session_state.lif_last_current = 0.0
        st.session_state.lif_reset_counter += 1 

        st.rerun()
    
    st.sidebar.header('Model Parameters')        
    st.sidebar.subheader('Current stimulus settings')

    i_amp = st.sidebar.slider('Current amplitude ($\mu A/cm^2$)', -1.0, 3.0, 0.0, 0.25, key=f'i_amp_{lif_reset_key}')
    
    return {
        'I_amp': i_amp,
        'Current_list': st.session_state.lif_current_list,
        'Frequency_list': st.session_state.lif_frequency_list,
        'Last_current': st.session_state.lif_last_current
    }

def prepare_lif_plots():
    
    params = create_sidebar_controls_lif()

    neuron = LIF()

    STEP_SIZE = 0.01  # ms
    SIMULATION_TIME = 200.0 # ms
    time = np.arange(0, SIMULATION_TIME, STEP_SIZE)

    i_amp = params['I_amp']
    i_start = 20.0
    i_end = SIMULATION_TIME - 20.0

    lif_current_list = st.session_state.lif_current_list
    frequency_list = st.session_state.lif_frequency_list
    lif_last_current = st.session_state.lif_last_current

    current_changed = abs(i_amp - lif_last_current) > 0.05
    current_exists = any(abs(c - i_amp) <= 0.05 for c in lif_current_list)
    
    current_stimulus = create_current_stimulus_array(time,
                                                    i_amp,
                                                    i_start,
                                                    i_end                                               
                                                    )
    
    solution = neuron.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
        
    v = solution[:, 0]
    
    if current_changed and not current_exists:

        with st.spinner(f"Running F-I simulation for {i_amp:.1f} $\mu A/cm^2$ ..."):

            spike_indices = []
            threshold = -50           
    
            solution = neuron.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
        
            v = solution[:, 0]

            for i in range(1, len(v)):
                if v[i-1] < threshold and v[i] >= threshold:
                    spike_indices.append(i)
            
            if(len(spike_indices) > 1):
                duration = i_end - i_start
                frequency = 1000 * len(spike_indices) / duration

            else:
                duration = 0
                frequency = 0
                        
            st.session_state.lif_current_list.append(i_amp)
            st.session_state.lif_frequency_list.append(frequency)

            sorted_pairs = sorted(zip(st.session_state.lif_current_list, st.session_state.lif_frequency_list))
            st.session_state.lif_current_list, st.session_state.lif_frequency_list = zip(*sorted_pairs) if sorted_pairs else ([], [])
            st.session_state.lif_current_list = list(st.session_state.lif_current_list)
            st.session_state.lif_frequency_list = list(st.session_state.lif_frequency_list)
            
            st.success(f"Added: {i_amp:.2f} $\mu A/cm^{2}$ → {frequency:.1f} Hz")
            #st.rerun()
            
    elif current_exists and current_changed:
        st.info(f"Current {i_amp:.1f} $\mu A/cm^2$ already tested")
    
    st.session_state.lif_last_current = i_amp
    
    return v, time, current_stimulus, st.session_state.lif_current_list, st.session_state.lif_frequency_list, st.session_state.lif_last_current

def display_lif_theory():
    with st.expander('About Leaky Integrate and Fire model'):
        st.markdown("""
    
    A simple model to explain the current vs frequency relationship as seen in many biological neurons. \
    This model only incorporates a leak channel that ensures the membrane potential of the neuron model \
    returns to it's resting state value in the absence of inputs.                    
    
    Using this setup, you can vary the input current's strength, as well as the time interval over which the current is applied. 
    The current in put in this case is pulse input, that can raise the membrane potential. As the membrane potential rises and crosses a threshold, 
    the model artificially is set to a value of 0 before being reset to it's resting value. In this way, this setup demonstrates neuron
    behavior without taking into account any biophysical mechanisms into account.
    """)

def create_sidebar_controls_hh():
    '''
    Sidebar controls for the Hodgkin Huxley (HH) neuron model

    Returns:

    params : dict
    Dictionary containing all parameter values for the HH model
    '''

    if 'hh_current_list' not in st.session_state:
        st.session_state.hh_current_list = []

    if 'hh_last_current' not in st.session_state:
        st.session_state.hh_last_current = 0.0

    if 'hh_frequency_list_control' not in st.session_state:
        st.session_state.hh_frequency_list_control = []

    if 'hh_frequency_list_alt' not in st.session_state:
        st.session_state.hh_frequency_list_alt = []

    if 'hh_reset_counter' not in st.session_state:
        st.session_state.hh_reset_counter = 0

    hh_reset_key = st.session_state.hh_reset_counter

    hh_reset_pressed = st.sidebar.button("Reset")
    
    if hh_reset_pressed:
        st.session_state.hh_current_list = []
        st.session_state.hh_frequency_list_control = []
        st.session_state.hh_frequency_list_alt = []
        st.session_state.hh_last_current = 0.0
        st.session_state.hh_reset_counter += 1 

        st.rerun()

    st.sidebar.header('Model Parameters')
    st.sidebar.subheader('Temperature and Q10 values')
    q_gate = st.sidebar.slider('Q10 for conformation dependent processes', 2.0, 4.0, 3.0, 0.1)
    q_cond = st.sidebar.slider('Q10 for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1)

    st.sidebar.subheader('Current stimulus settings')
    i_amp = st.sidebar.slider('Current amplitude ($\mu A/cm^2$)', -3.0, 15.0, 0.0, 0.5, format="%0.1f", key=f'i_amp_{hh_reset_key}')
    
    temperature = st.selectbox('Temperature in Celsius (control fixed at 6.3 Celsius)', [0.0, 10.0, -10.0], width=300) # control temperature = 6.3 celsius

    # Return all parameters as a dictionary
    return {
        'temperature': temperature,
        'Q_gate': q_gate,
        'Q_cond': q_cond,
        'I_amp': i_amp,
        'HH_Current_list': st.session_state.hh_current_list,
        'Frequency_list_control': st.session_state.hh_frequency_list_control,
        'Frequency_list_alt': st.session_state.hh_frequency_list_alt,
        'HH_Last_current': st.session_state.hh_last_current
    }

def prepare_hh_plots():
    
    params = create_sidebar_controls_hh()
    temperature = params['temperature']
    q_gate = params['Q_gate']
    q_cond = params['Q_cond']

    neuron_control = HodgkinHuxley(6.3, q_gate, q_cond)
    neuron_alt = HodgkinHuxley(temperature, q_gate, q_cond)

    STEP_SIZE = 0.01  # ms
    SIMULATION_TIME = 300.0 # ms
    time = np.arange(0, SIMULATION_TIME, STEP_SIZE)

    solution_alt = None
    solution_control = None
 
    i_amp = params['I_amp']
    i_start = 50.0
    i_end = SIMULATION_TIME - 50.0

    hh_current_list = st.session_state.hh_current_list
    frequency_list_control = st.session_state.hh_frequency_list_control
    frequency_list_alt = st.session_state.hh_frequency_list_alt
    hh_last_current = st.session_state.hh_last_current

    current_changed = abs(i_amp - hh_last_current) >= 0.5
    current_exists = any(abs(c - i_amp) == 0.00 for c in hh_current_list)
    
    current_stimulus = create_current_stimulus_array(time,
                                                    i_amp,
                                                    i_start,
                                                    i_end                                               
                                                    )
    
    solution_control = neuron_control.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
    
    solution_alt = neuron_alt.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)

    v = solution_control[:, 0]
    v_ = solution_alt[:, 0]

    if current_changed and not current_exists:

        with st.spinner(f"Running F-I simulation for {i_amp:.1f} $\mu A/cm^{2}$..."):
            spike_indices = []
            spike_indices_ = []
            threshold = -20          
    
            solution_control = neuron_control.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
            solution_alt = neuron_alt.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
        
            v = solution_control[:, 0]
            v_ = solution_alt[:, 0]

            for i in range(1, len(v)):
                if v[i-1] < threshold and v[i] >= threshold:
                    spike_indices.append(i)
                if v_[i-1] < threshold and v_[i] >= threshold:
                    spike_indices_.append(i)
            
            if len(spike_indices) > 0:
                duration = i_end - i_start
                frequency_control = 1000 * len(spike_indices) / duration
            else:
                frequency_control = 0

            if len(spike_indices_) > 0:
                duration_ = i_end - i_start
                frequency_alt = 1000 * len(spike_indices_) / duration_
            else:
                frequency_alt = 0

            st.session_state.hh_current_list.append(i_amp)
            st.session_state.hh_frequency_list_control.append(frequency_control)
            st.session_state.hh_frequency_list_alt.append(frequency_alt)

            sorted_pairs = sorted(zip(st.session_state.hh_current_list, st.session_state.hh_frequency_list_control, st.session_state.hh_frequency_list_alt))
            st.session_state.hh_current_list, st.session_state.hh_frequency_list_control, st.session_state.hh_frequency_list_alt = zip(*sorted_pairs)
            st.session_state.hh_current_list = list(st.session_state.hh_current_list)
            st.session_state.hh_frequency_list_control = list(st.session_state.hh_frequency_list_control)
            st.session_state.hh_frequency_list_alt = list(st.session_state.hh_frequency_list_alt)
            
            st.success(f"Added: {i_amp:.1f} $\mu A/cm^{2}$")
            #  st.rerun()

    
    else:
        st.info(f"Current {i_amp:.1f} $\mu A/cm^2$ already tested")

    st.session_state.hh_last_current = i_amp    

    return v, v_, time, current_stimulus, temperature, st.session_state.hh_current_list, st.session_state.hh_frequency_list_control, st.session_state.hh_frequency_list_alt, st.session_state.hh_last_current

def create_sidebar_controls_hvcra():
    '''
    Sidebar controls for the HVC(RA) projection neuron model

    Returns:

    params : dict
    Dictionary containing all parameter values for the HVC(RA)
    '''

    st.sidebar.header('Model Parameters')
    st.sidebar.subheader('Temperature and Q10 values')

    col1, col2 = st.columns(2)
    with col1:
        input_type = st.selectbox('Input type', ['Current input', 'Synaptic input'], width=200)
    with col2:
        temperature = st.selectbox('Altered Temperature in $^o$C (control set to 40$^o$ C)', [30.0, 35.0], width=300) #  control temperature = 40.0 celsius
    
    q_gate = st.sidebar.slider('Q10 for conformation dependent processes', 2.0, 4.0, 3.0, 0.1)
    q_cond = st.sidebar.slider('Q10 for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1)

    if input_type == 'Current input':
        
        if 'hvcra_current_input_list' not in st.session_state:
            st.session_state.hvcra_current_input_list = []
        if 'hvcra_frequency_control_list' not in st.session_state:
            st.session_state.hvcra_frequency_control_list = []
        if 'hvcra_frequency_alt_list' not in st.session_state:
            st.session_state.hvcra_frequency_alt_list = []
        if 'hvcra_last_current_input' not in st.session_state:
            st.session_state.hvcra_last_current_input = 0.0
        
        if 'hvcra_reset_counter' not in st.session_state:
            st.session_state.hvcra_reset_counter = 0
        
        hvcra_reset_key = st.session_state.hvcra_reset_counter
        hvcra_reset_pressed = st.sidebar.button("Reset")

        if hvcra_reset_pressed:
            st.session_state.hvcra_current_input_list = []
            st.session_state.hvcra_frequency_control_list = []
            st.session_state.hvcra_frequency_alt_list = []
            st.session_state.hvcra_last_current_input = 0.0
            st.session_state.hvcra_reset_counter += 1 

            st.rerun()

        st.sidebar.subheader('Current stimulus settings')

        i_amp = st.sidebar.slider('Current amplitude (units?)', -0.5, 1.0, 0.00, 0.05, key=f'i_amp_{hvcra_reset_key}')
        #i_start = st.sidebar.slider('Current start time (ms)', 150.0, 200.0, 150.0, 10.0, key=f'i_start_{hvcra_reset_key}')
        #i_end = st.sidebar.slider('Current end time (ms)', 190.0, 200.0, 190.0, 10.0,  key=f'i_end_{hvcra_reset_key}')

        return {
            'temperature': temperature,
            'Q_gate': q_gate,
            'Q_cond': q_cond,
            'I_amp': i_amp,
            #'I_start': i_start,
            #'I_end': i_end,
            'Input_type': input_type,
            'Current_input_list': st.session_state.hvcra_current_input_list,
            'Frequency_control_list': st.session_state.hvcra_frequency_control_list,
            'Frequency_alt_list': st.session_state.hvcra_frequency_alt_list,
            'Last_current': st.session_state.hvcra_last_current_input
            }
    
    if input_type == 'Synaptic input':

        if 'hvcra_synaptic_input_list' not in st.session_state:
            st.session_state.hvcra_synaptic_input_list = []
        if 'response_time_control_list' not in st.session_state:
            st.session_state.response_time_control_list = []
        if 'response_time_alt_list' not in st.session_state:
            st.session_state.response_time_alt_list = []
        if 'hvcra_last_synaptic_input' not in st.session_state:
            st.session_state.hvcra_last_synaptic_input = 0.0
        if 'hvcra_reset_counter' not in st.session_state:
            st.session_state.hvcra_reset_counter = 0

        hvcra_reset_key = st.session_state.hvcra_reset_counter

        hvcra_reset_pressed = st.sidebar.button("Reset")

        if hvcra_reset_pressed:
            st.session_state.hvcra_synaptic_input_list = []
            st.session_state.response_time_control_list = []
            st.session_state.response_time_alt_list = []
            st.session_state.hvcra_last_synaptic_input = 0.0
            st.session_state.hvcra_reset_counter += 1 

            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            external_input = st.selectbox('Add External Input', ['No', 'Yes'], width=200)
        with col2:
            noise_input = st.selectbox('Add Noise Input', ['No', 'Yes'], width=200)

        st.sidebar.subheader('Synaptic Input Settings')

        ge_max = st.sidebar.slider('Excitatory Synapse Strength (mS/cm^2)', 0.00, 1.0, 0.00, 0.01, key=f'ge_max_{hvcra_reset_key}')

        gi_max = st.sidebar.slider('inhibitory synapse strength (mS/cm^2)', 0.0, 0.5, 0.0, 0.05, key=f'gi_max_{hvcra_reset_key}')
        gi_start = st.sidebar.slider('inhibitory synapse start time (ms)', 100.0, 200.0, 150.0, 0.5, key=f'gi_start_{hvcra_reset_key}')

        if external_input == 'Yes' and noise_input == 'Yes':
            freq = st.sidebar.slider('external input frequency (Hz)', 500.0, 1500.0, 500.0, 250.0)
            external_input_strength = st.sidebar.slider('external input max kick (mS/cm^2)', 0.001, 0.005, 0.001, 0.001, format="%0.3f")
            freq_noise = st.sidebar.slider('noise input frequency (Hz)', 50.0, 200.0, 200.0, 50.0)
            noise_strength = st.sidebar.slider('noise input max kick (mS/cm^2)', 0.01, 0.06, 0.01, 0.01)
            
            return {
                'temperature': temperature,
                'Q_gate': q_gate,
                'Q_cond': q_cond,
                'Input_type': input_type,
                'ge_max': ge_max,
                'gi_max': gi_max,
                'gi_start': gi_start,
                'External_input': external_input,
                'Noise_input': noise_input,
                'freq': freq,
                'external_input_strength': external_input_strength,
                'freq_noise': freq_noise,
                'noise_strength': noise_strength,
                'Synaptic_input_list': st.session_state.synaptic_input_list,
                'Response_time_q_list': st.session_state.response_time_q_list,
                'Last_synaptic_input': st.session_state.last_synaptic_input
            }
            
        elif external_input == 'Yes' and noise_input == 'No':
            freq = st.sidebar.slider('external input frequency (Hz)', 500.0, 1500.0, 500.0, 250.0)
            external_input_strength = st.sidebar.slider('external input max kick (mS/cm^2)', 0.001, 0.005, 0.001, 0.001, format="%0.3f")
            
            return {
                'temperature': temperature,
                'Q_gate': q_gate,
                'Q_cond': q_cond,
                'Input_type': input_type,
                'ge_max': ge_max,
                'gi_max': gi_max,
                'gi_start': gi_start,
                'External_input': external_input,
                'Noise_input': noise_input,
                'freq': freq,
                'external_input_strength': external_input_strength,
                'Synaptic_input_list': st.session_state.synaptic_input_list,
                'Response_time_q_list': st.session_state.response_time_q_list,
                'Last_synaptic_input': st.session_state.last_synaptic_input
            }
            
        elif external_input == 'No' and noise_input == 'Yes':
            freq_noise = st.sidebar.slider('noise input frequency (Hz)', 100.0, 200.0, 200.0, 25.0)
            noise_strength = st.sidebar.slider('noise input max kick (mS/cm^2)', 0.01, 0.06, 0.01, 0.01)
            
            return {
                'temperature': temperature,
                'Q_gate': q_gate,
                'Q_cond': q_cond,
                'Input_type': input_type,
                'ge_max': ge_max,
                'gi_max': gi_max,
                'gi_start': gi_start,
                'External_input': external_input,
                'Noise_input': noise_input,
                'freq_noise': freq_noise,
                'noise_strength': noise_strength,
                'Synaptic_input_list': st.session_state.synaptic_input_list,
                'Response_time_q_list': st.session_state.response_time_q_list,
                'Last_synaptic_input': st.session_state.last_synaptic_input
            }
            
        else:
            
            return {
                'temperature': temperature,
                'Q_gate': q_gate,
                'Q_cond': q_cond,
                'Input_type': input_type,
                'ge_max': ge_max,
                'gi_max': gi_max,
                'gi_start': gi_start,
                'External_input': external_input,
                'Noise_input': noise_input,
                'Synaptic_input_list': st.session_state.hvcra_synaptic_input_list,
                'Response_time_control_list': st.session_state.response_time_control_list,
                'Response_time_alt_list': st.session_state.response_time_alt_list,                
                'Last_synaptic_input': st.session_state.hvcra_last_synaptic_input
            }

def create_sidebar_controls_hvci():
    '''
    Sidebar controls for the HVC Interneuron HVC(I) model

    Returns:

    params : dict
    Dictionary containing all parameter values for the HVC(I) model
    '''

    st.sidebar.header('Model Parameters')

    input_type = st.selectbox('Input type', ['Current input', 'Synaptic input'])

    st.sidebar.subheader('Temperature and Q10 values')
    temperature = st.selectbox('Temperature in Celsius', [30.0, 35.0]) #  control temperature = 40.0 celsius
    q_gate = st.sidebar.slider('Q10 for conformation dependent processes', 2.0, 4.0, 3.0, 0.1)
    q_cond = st.sidebar.slider('Q10 for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1)

    if input_type == 'Current input':

        if 'hvci_current_input_list' not in st.session_state:
            st.session_state.hvci_current_input_list = []
        if 'hvci_frequency_control_list' not in st.session_state:
            st.session_state.hvci_frequency_control_list = []
        if 'hvci_frequency_alt_list' not in st.session_state:
            st.session_state.hvci_frequency_alt_list = []
        if 'hvci_last_current_input' not in st.session_state:
            st.session_state.hvci_last_current_input = 0.0
        
        if 'hvci_reset_counter' not in st.session_state:
            st.session_state.hvci_reset_counter = 0
        
        hvci_reset_key = st.session_state.hvci_reset_counter
        hvci_reset_pressed = st.sidebar.button("Reset")

        if hvci_reset_pressed:
            st.session_state.hvci_current_input_list = []
            st.session_state.hvci_frequency_control_list = []
            st.session_state.hvci_frequency_alt_list = []
            st.session_state.hvci_last_current_input = 0.0
            st.session_state.hvci_reset_counter += 1 

            st.rerun()

        st.sidebar.subheader('Current stimulus settings')

        i_amp = st.sidebar.slider(f'Current amplitude ($\mu A/cm^{2}$)', -2.0, 20.0, 0.0, 1.0, key=f'i_amp_{hvci_reset_key}')
        
        return {
            'temperature': temperature,
            'Q_gate': q_gate,
            'Q_cond': q_cond,
            'I_amp': i_amp,
            'Input_type': input_type,
            'Current_input_list': st.session_state.hvci_current_input_list,
            'Frequency_control_list': st.session_state.hvci_frequency_control_list,
            'Frequency_alt_list': st.session_state.hvci_frequency_alt_list,
            'Last_current': st.session_state.hvci_last_current_input
        }
    
    if input_type == 'Synaptic input':

        noise_input = st.selectbox('Add noise input', ['No', 'Yes'])

        st.sidebar.subheader('synaptic input settings')

        ge_max = st.sidebar.slider('excitatory synapse strength (mS/cm^2)', 0.0, 1.0, 0.0, 0.05)
        ge_start = st.sidebar.slider('excitatory synapse start time (ms)', 50.0, 200.0, 150.0, 0.5)

        gi_max = st.sidebar.slider('inhibitory synapse strength (mS/cm^2)', 0.0, 0.5, 0.0, 0.05)
        gi_start = st.sidebar.slider('inhibitory synapse start time (ms)', 50.0, 200.0, 150.0, 0.5)

        if noise_input == 'Yes':
            freq_noise = st.sidebar.slider('noise input frequency (Hz)', 50.0, 300.0, 200.0, 50.0)
        
            # Return all parameters as a dictionary
            return {
                'temperature': temperature,
                'Q_gate': q_gate,
                'Q_cond': q_cond,
                'Input_type': input_type,
                'ge_max': ge_max,
                'ge_start': ge_start,
                'gi_max': gi_max,
                'gi_start': gi_start,
                'Noise_input': noise_input,
                'freq_noise': freq_noise
            }
        
        else:
            return {
                'temperature': temperature,
                'Q_gate': q_gate,
                'Q_cond': q_cond,
                'Input_type': input_type,
                'ge_max': ge_max,
                'ge_start': ge_start,
                'gi_max': gi_max,
                'gi_start': gi_start,
                'Noise_input': noise_input
            }

def display_hh_theory():
    with st.expander('About Hodgkin-Huxley model'):
        st.markdown("""
        ## About Hodgkin Huxley model
        
        A complex dynamical model that mimics the biophysical mechanisms involved in action potential generation.
        
        ### Ion channels incorporated
        - Sodium channel
        - Potassium channel
        - Leak Channel
        """)
        
        try:
            image = Image.open("../streamlit_images/HH.png")
            st.image(image, caption="HH model", use_container_width=True)
        except FileNotFoundError:
            st.info("Image not found")

def display_hvcra_theory():
    pass

def prepare_hvcra_plots():

    fluctuations = 'off'
    input_changed = 0
    input_exists = 0
    params = create_sidebar_controls_hvcra()
    temperature = params['temperature']
    q_gate = params['Q_gate']
    q_cond = params['Q_cond']

    neuron_control = HvcProjectionNeuron(40.0, q_gate, q_cond)
    neuron_alt = HvcProjectionNeuron(temperature, q_gate, q_cond)

    input_type = params['Input_type']

    STEP_SIZE = 0.01  # ms
    SIMULATION_TIME = 300.0 # ms
    time = np.arange(0, SIMULATION_TIME, STEP_SIZE)

    vs = np.full((30000, 1), 0)
    vs_ = np.full((30000, 1), 0)
    vd = np.full((30000, 1), 0)
    vd_ = np.full((30000, 1), 0)

    solution_alt = None
    solution_control = None

    response_time_displayed = None
    response_time_displayed_ = None
    response_time_q10 = None

    if input_type == 'Current input':

        i_amp = params['I_amp']
        i_start = 150.0
        i_end = 170.0
        
        current_stimulus = create_current_stimulus_array(time,
                                                        i_amp,
                                                        i_start,
                                                        i_end                                               
                                                        )
        
        solution_control = neuron_control.simulate(time,
                                                   STEP_SIZE,
                                                   current_stimulus_array=current_stimulus
                                                  )
        
        solution_alt = neuron_alt.simulate(time,
                                           STEP_SIZE,
                                           current_stimulus_array=current_stimulus)
        
        vs = solution_control[:, 1]
        vd = solution_control[:, 0]

        vs_ = solution_alt[:, 1]
        vd_ = solution_alt[:, 0]
        
        hvcra_current_input_list = st.session_state.hvcra_current_input_list
        hvcra_frequency_control_list = st.session_state.hvcra_frequency_control_list
        hvcra_frequency_alt_list = st.session_state.hvcra_frequency_alt_list
        hvcra_last_current = st.session_state.hvcra_last_current_input

        current_changed = abs(i_amp - hvcra_last_current) > 0.01
        current_exists = any(c == i_amp for c in hvcra_current_input_list)

        if current_changed and not current_exists:

            with st.spinner(f"Running simulation for {i_amp:.2f}..."):
                
                solution_control = neuron_control.simulate(time,
                                                        STEP_SIZE,
                                                        current_stimulus_array=current_stimulus
                                                        )
                
                solution_alt = neuron_alt.simulate(time,
                                                STEP_SIZE,
                                                current_stimulus_array=current_stimulus)

                vs = solution_control[:, 1]
                vd = solution_control[:, 0]

                vs_ = solution_alt[:, 1]
                vd_ = solution_alt[:, 0]

                st.success(f"finished running!")

                spike_count_control = 0
                spike_count_alt = 0
                threshold = -20           
        
                for i in range(1, len(vs)):
                    if vs[i-1] < threshold and vs[i] >= threshold:
                        spike_count_control += 1
                    if vs_[i-1] < threshold and vs_[i] >= threshold:
                        spike_count_alt += 1
                    
                frequency_control = spike_count_control
                frequency_alt = spike_count_alt

                st.session_state.hvcra_current_input_list.append(i_amp)
                st.session_state.hvcra_frequency_control_list.append(frequency_control)
                st.session_state.hvcra_frequency_alt_list.append(frequency_alt)
                st.session_state.hvcra_last_current_input = i_amp

                sorted_pairs = sorted(zip(st.session_state.hvcra_current_input_list, st.session_state.hvcra_frequency_control_list, st.session_state.hvcra_frequency_alt_list))
                st.session_state.hvcra_current_input_list, st.session_state.hvcra_frequency_control_list, st.session_state.hvcra_frequency_alt_list = zip(*sorted_pairs)
                st.session_state.hvcra_current_input_list = list(st.session_state.hvcra_current_input_list)
                st.session_state.hvcra_frequency_control_list = list(st.session_state.hvcra_frequency_control_list)
                st.session_state.hvcra_frequency_alt_list = list(st.session_state.hvcra_frequency_alt_list)
        
        elif current_exists and current_changed:
            st.info(f"Current input {i_amp:.2f} already tested")
        
        st.session_state.hvcra_last_current_input = i_amp

        return input_type, fluctuations, vs, vs_, vd, vd_, time, temperature, st.session_state.hvcra_current_input_list, st.session_state.hvcra_frequency_control_list, st.session_state.hvcra_frequency_alt_list, st.session_state.hvcra_last_current_input
 
    if input_type == 'Synaptic input':

        ge_max = params['ge_max']
        gi_max = params['gi_max']
        gi_start = params['gi_start']
        external_input = params['External_input']
        noise_input = params['Noise_input']
        
        ge_start = 150.0 
        noise_strength = None
        external_input_strength = None
        freq_noise = None
        freq = None

        if external_input == 'Yes' or noise_input == 'Yes':
            fluctuations = 'on'

        if external_input == 'Yes' and noise_input == 'Yes':
            freq = params['freq']
            freq_noise = params['freq_noise']  
            external_input_strength = params['external_input_strength']
            noise_strength = params['noise_strength']

        elif external_input == 'Yes' and noise_input == 'No':
            freq = params['freq']
            external_input_strength = params['external_input_strength']
    
        if external_input == 'No' and noise_input == 'Yes':
            freq_noise = params['freq_noise']
            noise_strength = params['noise_strength']

        excitatory_synapse_stimulus_control = create_synapse_stimulus_array(time,
                                                40.0,
                                                q_gate,
                                                ge_max,
                                                ge_start,
                                                STEP_SIZE
                                                )
        
        excitatory_synapse_stimulus_alt = create_synapse_stimulus_array(time,
                                                temperature,
                                                q_gate,
                                                ge_max,
                                                ge_start,
                                                STEP_SIZE
                                                )

        inhibitory_synapse_stimulus_control = create_synapse_stimulus_array(time,
                                                40.0,
                                                q_gate,
                                                gi_max,
                                                gi_start,
                                                STEP_SIZE
                                                )
        
        inhibitory_synapse_stimulus_alt = create_synapse_stimulus_array(time,
                                                temperature,
                                                q_gate,
                                                gi_max,
                                                gi_start,
                                                STEP_SIZE
                                                )

        solution_control = neuron_control.simulate(time,
                                            STEP_SIZE,                                        
                                            excitatory_synapse_stimulus_array=excitatory_synapse_stimulus_control,
                                            inhibitory_synapse_stimulus_array=inhibitory_synapse_stimulus_control,
                                            external_input_freq=freq,
                                            external_input_strength=external_input_strength,
                                            noise_freq=freq_noise,
                                            noise_strength=noise_strength)
        
        solution_alt = neuron_alt.simulate(time,
                                            STEP_SIZE,
                                            excitatory_synapse_stimulus_array=excitatory_synapse_stimulus_alt,
                                            inhibitory_synapse_stimulus_array=inhibitory_synapse_stimulus_alt,
                                            external_input_freq=freq,
                                            external_input_strength=external_input_strength,
                                            noise_freq=freq_noise,
                                            noise_strength=noise_strength)

        vs = solution_control[:, 1]
        vd = solution_control[:, 0]

        vs_ = solution_alt[:, 1]
        vd_ = solution_alt[:, 0]

        hvcra_synaptic_input_list = st.session_state.hvcra_synaptic_input_list
        response_time_control_list = st.session_state.response_time_control_list
        response_time_alt_list = st.session_state.response_time_alt_list
        hvcra_last_synaptic_input = st.session_state.hvcra_last_synaptic_input 

        input_changed = abs(ge_max - hvcra_last_synaptic_input) > 0.005
        input_exists = any(g == ge_max for g in hvcra_synaptic_input_list)

        if input_changed and not input_exists:

            with st.spinner(f"Running simulation for {ge_max:.2f} mS/cm^2..."):

                excitatory_synapse_stimulus_control = create_synapse_stimulus_array(time,
                                                                                    40.0,
                                                                                    q_gate,
                                                                                    ge_max,
                                                                                    ge_start,
                                                                                    STEP_SIZE
                                                                                    )
                
                excitatory_synapse_stimulus_alt = create_synapse_stimulus_array(time,
                                                                                temperature,
                                                                                q_gate,
                                                                                ge_max,
                                                                                ge_start,
                                                                                STEP_SIZE
                                                                                )
                
                inhibitory_synapse_stimulus_control = create_synapse_stimulus_array(time,
                                                                                    40.0,
                                                                                    q_gate,
                                                                                    gi_max,
                                                                                    gi_start,
                                                                                    STEP_SIZE
                                                                                    )
                
                inhibitory_synapse_stimulus_alt = create_synapse_stimulus_array(time,
                                                                                temperature,
                                                                                q_gate,
                                                                                gi_max,
                                                                                gi_start,
                                                                                STEP_SIZE
                                                                                )

                solution_control = neuron_control.simulate(time,
                                                           STEP_SIZE,                                        
                                                           excitatory_synapse_stimulus_array=excitatory_synapse_stimulus_control,
                                                           inhibitory_synapse_stimulus_array=inhibitory_synapse_stimulus_control,
                                                           external_input_freq=freq,
                                                           external_input_strength=external_input_strength,
                                                           noise_freq=freq_noise,
                                                           noise_strength=noise_strength)
                
                solution_alt = neuron_alt.simulate(time,
                                                   STEP_SIZE,
                                                   excitatory_synapse_stimulus_array=excitatory_synapse_stimulus_alt,
                                                   inhibitory_synapse_stimulus_array=inhibitory_synapse_stimulus_alt,
                                                   external_input_freq=freq,
                                                   external_input_strength=external_input_strength,
                                                   noise_freq=freq_noise,
                                                   noise_strength=noise_strength)
                
                vs = solution_control[:, 1]
                vd = solution_control[:, 0]

                vs_ = solution_alt[:, 1]
                vd_ = solution_alt[:, 0]

                response_time_displayed = response_time(time, 
                                                        vd, 
                                                        vs, 
                                                        excitatory_input_start_time=ge_start, 
                                                        excitatory_input_strength=ge_max)
                
                response_time_displayed_ = response_time(time, 
                                                        vd_, 
                                                        vs_, 
                                                        excitatory_input_start_time=ge_start, 
                                                        excitatory_input_strength=ge_max)
                
                st.session_state.hvcra_synaptic_input_list.append(ge_max)
                st.session_state.response_time_control_list.append(response_time_displayed)
                st.session_state.response_time_alt_list.append(response_time_displayed_)
                st.session_state.hvcra_last_synaptic_input = ge_max

                sorted_pairs = sorted(zip(st.session_state.hvcra_synaptic_input_list, st.session_state.response_time_control_list, st.session_state.response_time_alt_list))
                st.session_state.hvcra_synaptic_input_list, st.session_state.response_time_control_list, st.session_state.response_time_alt_list = zip(*sorted_pairs)
                st.session_state.hvcra_synaptic_input_list = list(st.session_state.hvcra_synaptic_input_list)
                st.session_state.response_time_control_list = list(st.session_state.response_time_control_list)
                st.session_state.response_time_alt_list = list(st.session_state.response_time_alt_list)


                st.success(f"finished running!")
            
        elif input_exists and input_changed:
            st.info(f"Synaptic input {ge_max:.2f} \mS /cm^{2} already tested")
            st.session_state.hvcra_last_synaptic_input = ge_max
            
        return input_type, fluctuations, vs, vs_, vd, vd_, time, temperature, st.session_state.hvcra_synaptic_input_list, st.session_state.response_time_control_list, st.session_state.response_time_alt_list, st.session_state.hvcra_last_synaptic_input

def display_hvci_theory():
    pass

def prepare_hvci_plots():
    
    params = create_sidebar_controls_hvci()
    temperature = params['temperature']
    q_gate = params['Q_gate']
    q_cond = params['Q_cond']

    neuron_control = HvcInterNeuron(40.0, q_gate, q_cond)
    neuron_alt = HvcInterNeuron(temperature, q_gate, q_cond)

    input_type = params['Input_type']

    STEP_SIZE = 0.01  # ms
    if input_type == 'Current input':
        simulation_time = 500.0 # ms
    else:
        simulation_time = 1000.0
    time = np.arange(0, simulation_time, STEP_SIZE)

    v = np.full((30000, 1), 0)
    v_ = np.full((30000, 1), 0)

    solution_alt = None
    solution_control = None
    current_stimulus = None
    if input_type == 'Current input':
    
        i_amp = params['I_amp']
        i_start = 50.0
        i_end = simulation_time - 50.0
        
        current_stimulus = create_current_stimulus_array(time,
                                                        i_amp,
                                                        i_start,
                                                        i_end                                               
                                                        )
        
        solution_control = neuron_control.simulate(time,
                                                            STEP_SIZE,
                                                            current_stimulus_array=current_stimulus
                                                            )
        
        solution_alt = neuron_alt.simulate(time,
                                                    STEP_SIZE,
                                                    current_stimulus_array=current_stimulus)
        
        v = solution_control[:, 0]
        v_ = solution_alt[:, 0]
        
        hvci_current_input_list = st.session_state.hvci_current_input_list
        hvci_frequency_control_list = st.session_state.hvci_frequency_control_list
        hvci_frequency_alt_list = st.session_state.hvci_frequency_alt_list
        hvci_last_current = st.session_state.hvci_last_current_input

        current_changed = abs(i_amp - hvci_last_current) > 0.01
        current_exists = any(c == i_amp for c in hvci_current_input_list)

        if current_changed and not current_exists:

            with st.spinner(f"Running simulation for {i_amp:.2f}..."):

                current_stimulus = create_current_stimulus_array(time,
                                                                i_amp,
                                                                i_start,
                                                                i_end                                               
                                                                )
                
                solution_control = neuron_control.simulate(time,
                                                        STEP_SIZE,
                                                        current_stimulus_array=current_stimulus
                                                        )
                
                solution_alt = neuron_alt.simulate(time,
                                                STEP_SIZE,
                                                current_stimulus_array=current_stimulus)

                st.success(f"finished running!")

                v = solution_control[:, 0]
                v_ = solution_alt[:, 0]

                spike_count_control = 0
                spike_count_alt = 0
                threshold = -20           
        
                for i in range(1, len(v)):
                    if v[i-1] < threshold and v[i] >= threshold:
                        spike_count_control += 1
                    if v_[i-1] < threshold and v_[i] >= threshold:
                        spike_count_alt += 1
                    
                frequency_control = spike_count_control
                frequency_alt = spike_count_alt

                st.session_state.hvci_current_input_list.append(i_amp)
                st.session_state.hvci_frequency_control_list.append(frequency_control)
                st.session_state.hvci_frequency_alt_list.append(frequency_alt)
                st.session_state.hvci_last_current_input = i_amp

                sorted_pairs = sorted(zip(st.session_state.hvci_current_input_list, st.session_state.hvci_frequency_control_list, st.session_state.hvci_frequency_alt_list))
                st.session_state.hvci_current_input_list, st.session_state.hvci_frequency_control_list, st.session_state.hvci_frequency_alt_list = zip(*sorted_pairs)
                st.session_state.hvci_current_input_list = list(st.session_state.hvci_current_input_list)
                st.session_state.hvci_frequency_control_list = list(st.session_state.hvci_frequency_control_list)
                st.session_state.hvci_frequency_alt_list = list(st.session_state.hvci_frequency_alt_list)
            
        elif current_exists and current_changed:
            st.info(f"Current input {i_amp:.2f} already tested")
        
        st.session_state.hvci_last_current_input = i_amp
        
        return v, v_, time, current_stimulus, temperature, st.session_state.hvci_current_input_list, st.session_state.hvci_frequency_control_list, st.session_state.hvci_frequency_alt_list, st.session_state.hvci_last_current_input

    if input_type == 'Synaptic input':
        
        ge_max = params['ge_max']
        ge_start = params['ge_start']
        gi_max = params['gi_max']
        gi_start = params['gi_start']
        noise_input = params['Noise_input']

        noise_strength = None
        freq_noise = None

        if noise_input == 'Yes':
            freq_noise = params['freq_noise']  
            noise_strength = 0.45    
        
        excitatory_synapse_stimulus_control = create_synapse_stimulus_array(time,
                                                40.0,
                                                q_gate,
                                                ge_max,
                                                ge_start,
                                                STEP_SIZE
                                                )
        
        excitatory_synapse_stimulus_alt = create_synapse_stimulus_array(time,
                                                temperature,
                                                q_gate,
                                                ge_max,
                                                ge_start,
                                                STEP_SIZE
                                                )
        
        inhibitory_synapse_stimulus_control = create_synapse_stimulus_array(time,
                                                40.0,
                                                q_gate,
                                                gi_max,
                                                gi_start,
                                                STEP_SIZE
                                                )
        
        inhibitory_synapse_stimulus_alt = create_synapse_stimulus_array(time,
                                                temperature,
                                                q_gate,
                                                gi_max,
                                                gi_start,
                                                STEP_SIZE
                                                )
        
        solution_control = neuron_control.simulate(time,
                                            STEP_SIZE,                                        
                                            excitatory_synapse_stimulus_array=excitatory_synapse_stimulus_control,
                                            inhibitory_synapse_stimulus_array=inhibitory_synapse_stimulus_control,
                                            noise_freq=freq_noise,
                                            noise_strength=noise_strength)
        
        solution_alt = neuron_alt.simulate(time,
                                            STEP_SIZE,
                                            excitatory_synapse_stimulus_array=excitatory_synapse_stimulus_alt,
                                            inhibitory_synapse_stimulus_array=inhibitory_synapse_stimulus_alt,
                                            noise_freq=freq_noise,
                                            noise_strength=noise_strength)
        

