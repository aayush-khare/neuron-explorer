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
       #else:
        #    return None
    
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

    if 'current_list' not in st.session_state:
        st.session_state.current_list = [0.0]  # Start with 0 current
        st.session_state.frequency_list = [0.0]  # Start with 0 frequency
        st.session_state.last_current = 0.0
    
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0

    st.sidebar.header('Model Parameters')
        
    st.sidebar.subheader('Current stimulus settings')

    reset_key = st.session_state.reset_counter

    i_amp = st.sidebar.slider('Current amplitude (units?)', -3.0, 3.0, 0.0, 0.1, 
                             key=f'i_amp_{reset_key}')
    i_start = st.sidebar.slider('Current start time (ms)', 10.0, 100.0, 10.0, 10.0,
                               key=f'i_start_{reset_key}')
    i_end = st.sidebar.slider('Current end time (ms)', 90.0, 190.0, 90.0, 10.0,
                             key=f'i_end_{reset_key}')
    reset_pressed = st.sidebar.button("Reset F-I Data")
    
    if reset_pressed:
        st.session_state.current_list = [0.0]
        st.session_state.frequency_list = [0.0]
        st.session_state.last_current = 0.0
        st.session_state.reset_counter += 1 

        st.rerun()

    
    #i_amp = st.sidebar.slider('Current amplitude (units?)', -3.0, 3.0, 0.0, 0.1)
    #i_start = st.sidebar.slider('Current start time (ms)', 10.0, 100.0, 10.0, 10.0)
    #i_end = st.sidebar.slider('Current end time (ms)', 90.0, 190.0, 90.0, 10.0)
    
    
    
    #if st.sidebar.button("Reset F-I Data"):
    #    st.session_state.current_list = []
    #    st.session_state.frequency_list = []
    #    st.session_state.last_current = 0.0
    
    #    st.rerun()
    
    # Return all parameters as a dictionary
    return {
        'I_amp': i_amp,
        'I_start': i_start,
        'I_end': i_end,
        'Current_list': st.session_state.current_list,
        'Frequency_list': st.session_state.frequency_list,
        'Last_current': st.session_state.last_current
    }

def prepare_lif_plots():
    
    params = create_sidebar_controls_lif()

    neuron = LIF()

    STEP_SIZE = 0.01  # ms
    SIMULATION_TIME = 200.0 # ms
    time = np.arange(0, SIMULATION_TIME, STEP_SIZE)

    i_amp = params['I_amp']
    i_start = params['I_start']
    i_end = params['I_end']
    current_list = params['Current_list']
    frequency_list = params['Frequency_list']
    st.session_state.last_current = params['Last_current']

    current_changed = abs(i_amp - st.session_state.last_current) > 0.05
    current_exists = any(c == i_amp for c in current_list)
    
    current_stimulus = create_current_stimulus_array(time,
                                                    i_amp,
                                                    i_start,
                                                    i_end                                               
                                                    )
    
    solution = neuron.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
        
    v = solution[:, 0]
    
    if current_changed and not current_exists:

        with st.spinner(f"Running F-I simulation for {i_amp:.1f} ..."):

            spike_indices = []
            threshold = -50  # mV, adjust based on your model            
    
            solution = neuron.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
        
            v = solution[:, 0]

            for i in range(1, len(v)):
                if v[i-1] < threshold and v[i] >= threshold:
                    spike_indices.append(i)
            
            current_input_duration = i_end - i_start
            # Calculate frequency
            frequency = 1000 * len(spike_indices) / current_input_duration if current_input_duration  > 0 else 0
            
            current_list.append(i_amp)
            frequency_list.append(frequency)
            st.session_state.last_current = i_amp

            sorted_pairs = sorted(zip(current_list, frequency_list))
            current_list, frequency_list = zip(*sorted_pairs)
            current_list = list(current_list)
            frequency_list = list(frequency_list)
            
            st.success(f"Added: {i_amp:.1f} → {frequency:.1f} Hz")
            st.rerun()
            
    elif current_exists and current_changed:
        st.info(f"Current {i_amp:.1f} nA already tested")
        st.session_state.last_current = i_amp
    
    return v, time, current_stimulus, current_list, frequency_list, st.session_state.last_current

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

    st.sidebar.header('Model Parameters')

    input_type = st.selectbox('Input type', ['Current input'])

    # Temperature
    st.sidebar.subheader('Temperature and Q10 values')
    temperature = st.selectbox('Temperature in Celsius', [0.0, 10.0, -10.0]) # control temperature = 6.3 celsius
    q_gate = st.sidebar.slider('Q10 for conformation dependent processes', 2.0, 4.0, 3.0, 0.1)
    q_cond = st.sidebar.slider('Q10 for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1)

    # Current stimulus parameters
    if input_type == 'Current input':
        st.sidebar.subheader('Current stimulus settings')

        i_amp = st.sidebar.slider('Current amplitude (units?)', -30.0, 30.0, 0.0, 0.5)
        i_start = st.sidebar.slider('Current start time (ms)', 100.0, 150.0, 100.0, 10.0)
        i_end = st.sidebar.slider('Current end time (ms)', 150.0, 200.0, 160.0, 10.0)

        # Return all parameters as a dictionary
        return {
            'temperature': temperature,
            'Q_gate': q_gate,
            'Q_cond': q_cond,
            'I_amp': i_amp,
            'I_start': i_start,
            'I_end': i_end,
            'Input_type': input_type,
        }

def create_sidebar_controls_hvcra():
    '''
    Sidebar controls for the HVC(RA) projection neuron model

    Returns:

    params : dict
    Dictionary containing all parameter values for the HVC(RA)
    '''

    st.sidebar.header('Model Parameters')

    input_type = st.selectbox('Input type', ['Current input', 'Synaptic input'])

    # Temperature
    st.sidebar.subheader('Temperature and Q10 values')
    temperature = st.selectbox('Temperature in Celsius', [30.0, 35.0]) #  control temperature = 40.0 celsius
    q_gate = st.sidebar.slider('Q10 for conformation dependent processes', 2.0, 4.0, 3.0, 0.1)
    q_cond = st.sidebar.slider('Q10 for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1)

    # Current stimulus parameters
    if input_type == 'Current input':
        st.sidebar.subheader('Current stimulus settings')

        i_amp = st.sidebar.slider('Current amplitude (units?)', -5.0, 5.0, 0.0, 0.5)
        i_start = st.sidebar.slider('Current start time (ms)', 150.0, 200.0, 150.0, 10.0)
        i_end = st.sidebar.slider('Current end time (ms)', 150.0, 200.0, 150.0, 10.0)

        # Return all parameters as a dictionary
        return {
            'temperature': temperature,
            'Q_gate': q_gate,
            'Q_cond': q_cond,
            'I_amp': i_amp,
            'I_start': i_start,
            'I_end': i_end,
            'Input_type': input_type
            }
    
    if input_type == 'Synaptic input':

        external_input = st.selectbox('Add external input', ['No', 'Yes'])
        noise_input = st.selectbox('Add noise input', ['No', 'Yes'])

    # synaptic input parameters
        st.sidebar.subheader('synaptic input settings')

        ge_max = st.sidebar.slider('excitatory synapse strength (mS/cm^2)', 0.0, 1.0, 0.0, 0.05)
        ge_start = st.sidebar.slider('excitatory synapse start time (ms)', 150.0, 200.0, 150.0, 0.5)

        gi_max = st.sidebar.slider('inhibitory synapse strength (mS/cm^2)', 0.0, 0.5, 0.0, 0.05)
        gi_start = st.sidebar.slider('inhibitory synapse start time (ms)', 100.0, 200.0, 150.0, 0.5)

        if external_input == 'Yes' and noise_input == 'Yes':
            freq = st.sidebar.slider('external input frequency (Hz)', 500.0, 1500.0, 500.0, 250.0)
            external_input_strength = st.sidebar.slider('external input max kick (mS/cm^2)', 0.001, 0.005, 0.001, 0.001, format="%0.3f")
            freq_noise = st.sidebar.slider('noise input frequency (Hz)', 50.0, 200.0, 200.0, 50.0)
            noise_strength = st.sidebar.slider('noise input max kick (mS/cm^2)', 0.01, 0.06, 0.01, 0.01)

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
                'External_input': external_input,
                'Noise_input': noise_input,
                'freq': freq,
                'external_input_strength': external_input_strength,
                'freq_noise': freq_noise,
                'noise_strength': noise_strength
            }
        
        elif external_input == 'Yes' and noise_input == 'No':
            freq = st.sidebar.slider('external input frequency (Hz)', 500.0, 1500.0, 500.0, 250.0)
            external_input_strength = st.sidebar.slider('external input max kick (mS/cm^2)', 0.001, 0.005, 0.001, 0.001, format="%0.3f")
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
                'External_input': external_input,
                'Noise_input': noise_input,
                'freq': freq,
                'external_input_strength': external_input_strength
            }
        
        elif external_input == 'No' and noise_input == 'Yes':
            freq_noise = st.sidebar.slider('noise input frequency (Hz)', 100.0, 200.0, 200.0, 25.0)
            noise_strength = st.sidebar.slider('noise input max kick (mS/cm^2)', 0.01, 0.06, 0.01, 0.01)
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
                'External_input': external_input,
                'Noise_input': noise_input,
                'freq_noise': freq_noise,
                'noise_strength': noise_strength
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
                'External_input': external_input,
                'Noise_input': noise_input
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
        st.sidebar.subheader('Current stimulus settings')

        i_amp = st.sidebar.slider('Current amplitude (units?)', -30.0, 30.0, 0.0, 0.5)
        i_start = st.sidebar.slider('Current start time (ms)', 0.0, 50.0, 50.0, 10.0)
        i_end = st.sidebar.slider('Current end time (ms)', 50.0, 150.0, 60.0, 10.0)

        return {
            'temperature': temperature,
            'Q_gate': q_gate,
            'Q_cond': q_cond,
            'I_amp': i_amp,
            'I_start': i_start,
            'I_end': i_end,
            'Input_type': input_type,
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

def prepare_hh_plots():
    
    params = create_sidebar_controls_hh()
    temperature = params['temperature']
    q_gate = params['Q_gate']
    q_cond = params['Q_cond']

    neuron_control = HodgkinHuxley(6.3, q_gate, q_cond)
    neuron_alt = HodgkinHuxley(temperature, q_gate, q_cond)

    input_type = params['Input_type']

    STEP_SIZE = 0.01  # ms
    SIMULATION_TIME = 300.0 # ms
    time = np.arange(0, SIMULATION_TIME, STEP_SIZE)

    solution_alt = None
    solution_control = None

    if input_type == 'Current input':
    
        i_amp = params['I_amp']
        i_start = params['I_start']
        i_end = params['I_end']
        
        current_stimulus = create_current_stimulus_array(time,
                                                        i_amp,
                                                        i_start,
                                                        i_end                                               
                                                        )
        
        solution_control = neuron_control.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
        
        solution_alt = neuron_alt.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)

    v = solution_control[:, 0]
    v_ = solution_alt[:, 0]

    return v, v_, time, current_stimulus, temperature

def display_hvcra_theory():
    pass

def prepare_hvcra_plots():

    params = create_sidebar_controls_hvcra()
    temperature = params['temperature']
    q_gate = params['Q_gate']
    q_cond = params['Q_cond']

    neuron_control = HvcProjectionNeuron(40.0, q_gate, q_cond)
    neuron_alt = HvcProjectionNeuron(temperature, q_gate, q_cond)

    input_type = params['Input_type']#, ['Current input', 'Synaptic input']]

    STEP_SIZE = 0.01  # ms
    SIMULATION_TIME = 300.0 # ms
    time = np.arange(0, SIMULATION_TIME, STEP_SIZE)

    solution_alt = None
    solution_control = None

    response_time_displayed = None
    response_time_displayed_ = None

    if input_type == 'Current input':
    
        i_amp = params['I_amp']
        i_start = params['I_start']
        i_end = params['I_end']
        
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


    if input_type == 'Synaptic input':

        ge_max = params['ge_max']
        ge_start = params['ge_start']
        gi_max = params['gi_max']
        gi_start = params['gi_start']
        external_input = params['External_input']
        noise_input = params['Noise_input']

        noise_strength = None
        external_input_strength = None
        freq_noise = None
        freq = None

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

#    if input_type == 'Current input':
#        response_time_displayed = response_time(time, 
#                                                Vd, 
#                                                Vs, 
#                                                current_amplitude=I_amp, 
#                                                current_input_start_time=I_start)
#        response_time_displayed_ = response_time(time, 
#                                                 Vd_, 
#                                                 Vs_, 
#                                                 current_amplitude=I_amp, 
#                                                 current_input_start_time=I_start)

    if input_type == 'Synaptic input':
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
    
    response_time_q10 = None

    if response_time_displayed_ is not None and response_time_displayed is not None:
        response_time_q10 = (response_time_displayed_ / response_time_displayed) ** (0.1 * (40.0 - temperature))
    

    return vs, vs_, vd, vd_, time, temperature, response_time_displayed, response_time_displayed_, response_time_q10

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
    SIMULATION_TIME = 1000.0 # ms
    time = np.arange(0, SIMULATION_TIME, STEP_SIZE)

    solution_alt = None
    solution_control = None
    current_stimulus = None
    if input_type == 'Current input':
    
        i_amp = params['I_amp']
        i_start = params['I_start']
        i_end = params['I_end']
        
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
        
    v = solution_control[:, 0]
    v_ = solution_alt[:, 0]

    return v, v_, time, current_stimulus