import numpy as np
from math import exp, log
import streamlit as st
from PIL import Image
import os

from models.leaky_integrate_and_fire import LIF
from models.hodgkin_huxley import HodgkinHuxley
from models.hvc_projection_neuron import HvcProjectionNeuron
from models.hvc_interneuron import HvcInterNeuron

base_dir = os.path.dirname(__file__)
image_dir = os.path.join(base_dir, "images")

def display_introduction():
    
    st.markdown('<p class="big-font"> Neurons are special type of cells that generate electrical signals and communicate with each other' \
                ' through electrochemical processes triggered by these electrical signals. The generation of the electrical signal is '\
                ' governed by the flow of different types of ions across the cell membrane. Neurons are the fundamental unit of our nervous '\
                ' system. Developing an understanding of the nervous system in turn requires understanding how this fundamental unit functions. '\
                ' </p>', unsafe_allow_html=True) \
    
    neuron_image = os.path.join(image_dir, "neuron.jpg")
    st.image(neuron_image, width=1000)
        
    st.markdown('<p class="big-font"> In this interactive tool, you get to explore some models for simulating the dynamics of a neuron cell. \
                We will start by first developing a basic understanding of the electrical properties of neuron cells. With that, we will then \
                discuss how these electrical properties can be simulated on a computer. We wil start with the Leaky-Integrate-and-Fire (LIF) \
                neuron model. This is a simple model, yet captures effectively the most basic function of neuron cells, that is, accumulating \
                input signals, and if the inputs cross a threshold, emitting an action potential. This model does not incorporate any complex \
                biophysical details, but still gives a good insight into neuron function. We will build upon this insight by next looking at the \
                Hodgkin-Huxley model. This model incorporates mathematical formalisms to represent the complex biophysical mechanisms involved in \
                the movement of different ions across the cell membrane, and how these govern action potential generation. These mechanisms are \
                temperature dependent, and so we will also look into how a change in the surrounding temperatures affect the resulting dynamics of \
                a neuron. Finally you get to explore generalized biophysical models for the excitatory and inhibitory classes of neurons found in a \
                brain region called HVC (known as proper name) in songbird species.</p>', unsafe_allow_html=True)

def display_electrical_properties():

    st.markdown("                                      ")
    st.markdown("### Parts of a neuron")

    image_folder = os.path.join(image_dir, "neuron_parts")
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png'))])

    descriptions = [
    "A Neuron cell has a complex morphology with different parts contributing to its electrical properties.",
    "A neuron receives information from other neurons through the Dendrite, a structure that extends from the soma. The Dendrite has extensive branches, often referred \n"
    "to as the Dendritic tree. It is this Dendritic tree where a neuron receives majority of its inputs from other neurons.",
    "The cell body of a neuron is called the Soma which contains the nucleus. The soma can also receive inputs from other neurons, and is the site of primary electrical signal generation. \n"
    "This electrical signal, also termed as the action potential, is characterized by a large in and out flow of specific ions in a very short time interval.",
    "This action potential is then transmitted along a long cable like structure called the axon, that emerges out of the soma. The ends of the axon, called the axon terminals are sites \n"
    "which contain special chemicals called neurotransmitters. When an action potential arrives at the terminals, it can trigger specific processes that result in the release of these \n"
    "neurotransmitters. It is these neurotransmitters that are then captured at the dendrites of other neurons, triggering electrical activity in them and continuing the flow of information."
    ]

    max_index_neuron = len(descriptions) - 1
    min_index_neuron = 0

    if 'img_index' not in st.session_state:
        st.session_state.img_index = 0

    current_index = st.session_state.img_index
    show_next = current_index < max_index_neuron
    show_prev = current_index > min_index_neuron

    current_image = Image.open(os.path.join(image_folder, image_files[current_index]))
            
    col1, col2 = st.columns([1, 5])
    with col1:
        if show_prev:
            if st.button("Prev ") and current_index > 0:
                st.session_state.img_index -= 1
                current_index = st.session_state.img_index
                current_image = Image.open(os.path.join(image_folder, image_files[current_index]))

    with col2:
        if show_next:
            if st.button("Next ") and current_index < len(image_files) - 1:
                st.session_state.img_index += 1
                current_index = st.session_state.img_index
                current_image = Image.open(os.path.join(image_folder, image_files[current_index]))

    st.image(current_image, width=400)
    st.markdown(f"{descriptions[current_index]}")

    st.markdown("                                      ")
    st.markdown("### Action potential generation in a neuron")
    image_folder_ = os.path.join(image_dir, "neuron_dynamics")
    image_files_ = sorted([f for f in os.listdir(image_folder_) if f.endswith(('.png'))])

    descriptions_ = [
    "The electrical properties of a neuron can be attributed to a difference in the potential on either side of the cell membrane. This potential is maintained in the resting conditions \n"
    "such that the inside of the cell is at a negative potential with respect to the outside environment. This potential difference is known as the membrane potential of the neuron, and \n"
    "denotes the state of the neuron.",
    "There is a concentration gradient of ions of different types across the membrane. The membrane is typically impermeable to the flow of these ions. The membrane contains various ion \n"
    "channels that allow for the movement of ions. These channels are selective for specific types of ions. These channels open and close, a process dependent on change in the conformational \n"
    "states of the proteins making up the channel. The rates at which these conformation states change, and in turn result in the opening or closing of the channels are dependent on the membrane \n"
    "potential. ",
    "An excitatory input or injected current serves to elevate the membrane potential. This process is called membrane depolarization.",
    "As the membrane depolarizes, ion channels permeable to sodium ions open, allowing an influx of sodium ions that further aids depolarization. At a certain membrane potential called \n"
    "the threshold potential, a large positive feedback loop emerges, resulting in rapid depolarization",
    "The rapid depolarization subsequently opens voltage-gated potassium channels. The resulting efflux of potassium ions repolarizes the membrane back toward its resting potential.",
    "The rapid depolarization followed by repolarization which typically occurs within a millisecond in most neurons is known as an action potential. It is this sharp pulse that is \n"
    "generated at the region near where the axon originates from the soma, and is carried along the axon."
    ]

    max_index_dynamics = len(descriptions_) - 1
    min_index_dynamics = 0

    if 'img_index_' not in st.session_state:
        st.session_state.img_index_ = 0

    current_index_ = st.session_state.img_index_
    show_next_ = current_index_ < max_index_dynamics
    show_prev_ = current_index_ > min_index_dynamics

    current_image_ = Image.open(os.path.join(image_folder_, image_files_[current_index_]))

    col1, col2 = st.columns([1, 2])
    with col1:
        if show_prev_:
            if st.button("Prev") and current_index_ > 0:
                st.session_state.img_index_ -= 1
                current_index_ = st.session_state.img_index_
                current_image_ = Image.open(os.path.join(image_folder_, image_files_[current_index_]))
    
    with col2:
        if show_next_:
            if st.button("Next") and current_index_ < len(image_files_) - 1:
                st.session_state.img_index_ += 1
                current_index_ = st.session_state.img_index_
                current_image_ = Image.open(os.path.join(image_folder_, image_files_[current_index_]))

    st.image(current_image_, width=700)
    st.markdown(f"{descriptions_[current_index_]}")
    
    st.markdown("                                      ")
    st.markdown("### Biophysics behing modeling the dynamics of a neuron")

def display_temperature_properties():

    st.markdown("                                      ")
    st.markdown("### Temperature dependence of biophysical mechanisms governing neural dynamics")

    image_folder = os.path.join(image_dir, "temperature")
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png'))])

    descriptions = [
        'The movement of ions through the cellular medium is governed by a process called diffusion. This process affects the various conductances involved in the ion channel currents.'
        'The temperature dependence of this process is weak, quantifies by a $Q_{10}$ value typically in the range 1.2 - 1.4',
        'The conformational changes governing the opening and closing rates of ion channels are strongly affected by temperature changes. There temperature dependence can be quantified'
        'by $Q_{10}$ values typically in the range 2 - 4.'
    ]

    max_index_neuron = len(descriptions) - 1
    min_index_neuron = 0

    if 'img_index' not in st.session_state:
        st.session_state.img_index = 0

    current_index = st.session_state.img_index
    show_next = current_index < max_index_neuron
    show_prev = current_index > min_index_neuron

    current_image = Image.open(os.path.join(image_folder, image_files[current_index]))
            
    col1, col2 = st.columns([1, 4])
    with col1:
        if show_prev:
            if st.button("Prev ") and current_index > 0:
                st.session_state.img_index -= 1
                current_index = st.session_state.img_index
                current_image = Image.open(os.path.join(image_folder, image_files[current_index]))

    with col2:
        if show_next:
            if st.button("Next ") and current_index < len(image_files) - 1:
                st.session_state.img_index += 1
                current_index = st.session_state.img_index
                current_image = Image.open(os.path.join(image_folder, image_files[current_index]))

    st.image(current_image, width=800)
    st.markdown(f"{descriptions[current_index]}")

    st.markdown("                                      ")

def display_lif_theory():
    """
    Display the theoretical background of the Leaky Integrate and Fire model
    """

    st.markdown("""
    <style>
    .streamlit-expanderHeader {
        font-size: 150px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.expander('**About the Leaky Integrate and Fire Model**'):
        st.markdown(f"""
    
    The Leaky Integrate and Fire (LIF) model is a simple mathematical model that effectively \
    explains the most basic neuron function, i.e., integrating inputs to produce an output \
    spike. 

                    
    The Leaky in the LIF model corresponds to the incorporation of a leaky ion channel \
    in the model, that ensures that the membrane potential exponentially decays back to the \
    resting-state value in the absence of any inputs. This is the only ion channel incorporated \
    in the model. The membrane potential dynamics of the model are described by the equation """)
        
        st.latex(r''' C\frac{dV}{dt} = I_L + I_{inj} ''')

        st.markdown(f"""Here the first term on RHS is the leak current given by""")

        st.latex(r''' I_L = g_L(E_L - V)''')

        st.markdown(f""" In the presence of a constant current input, the differential equation has an analytical solution for the time evolution of the membrane potential given by""")

        st.latex(r''' V(t) = E_L + I_{inj}R + (V(0) - E_L - I_{inj}R)e^{-\frac{t}{\tau}}''')

        st.markdown(f""" Here $R = \\frac{{{1}}}{{g_L}}$  and the time constant $\\tau = RC$. 
                    This equation, holding for any value of the membrane potential below the threshold
                    value, can be used to obtain a response time for the LIF model. This response time, say
                    $t'$ corresponds to the time it takes for the neuron to fire an action potential from the
                    time at which the current pulse starts. Setting $V(0) = E_L$ and $V(t') = V_{{th}}$, the threshold
                    membrane potential, we get""")
        
        st.latex(r''' t' = \tau ln(\frac{I_{inj}R}{I_{inj}R + V(0) - V_{th}})''')

        st.markdown(f"""In the following setup, we have defined the threshold to be -50 mV. Each time the membrane potential crosses the threshold, we artificially set the membrane potential back to a reset value
                    which in this case is the same as the leak reversal potential. If the current pulse is still active, it would again depolarize the membrane potential towards the threshold. 
                    This means we can get an expression for the response time, and in turn the firing rate the LIF model will exhibit and its dependence on model parameters.""")

        st.markdown(f"""As can be seen, increasing the current can increase the firing rate. Additionally, it should be noted that the closer the initial value is to threshold membrane potential
                    the higher would be the firing rate, or faster the response.
    """)

def display_hh_theory():
    """
    Display the theoretical background of the Hodgkin-Huxley model.
    """

    with st.expander('About Hodgkin-Huxley model'):
        st.markdown("""
        ## About Hodgkin Huxley (HH) model
        
        Hodgkin Huxley model is a mathematical model that approximates the biophysical mechanisms involved in action potential generation. \
        Developed in 1952 by Alan Hodgkin and Andrew Huxley by studying the ionic mechanisms behind axon potential generation and propagation \
        in the squid giant axon, this model forms a basis for biophysical models describing neural dynamics for different classes of neurons, 
        and earned them the Nobel Prize in Physiology or Medicince.
        The HH model treats the cell membrane as a Capacitor, and voltage-dependent ion channels as variable resistors (variable as the resistance 
        or conductance at any instance depends on the membrane potential at that instant), that impart non-linearity to the model dynamics. \
        The mathematical form for the current flowing through the membrane is described as """)

        st.latex(r''' C\frac{dV}{dt} = I_{Na} + I_{K} + I_L + I_{inj} ''')

        st.markdown("""$I_{L}$ is the leak current and follows the same form as that in the LIF model. The sodium and potassium channel currents are described by """)

        st.latex(r''' I_{Na} = g_{Na}m^3h(V_{Na} - V)''')

        st.latex(r''' I_K = g_Kn^4(V_K - V)''')

        st.markdown(""" $g_{Na}$ and $g_K$ are fixed and are determined by the surface density of the respective channels on the membrane. The variability in the ion channel 
                    conductances arises from the fraction of the channels that are either open or closed. This is governed by the dynamics of the gating variables $n$, $m$ and $h$, 
                    which follow first order kinetics""")
        
        st.latex(r''' \frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)x''')

        st.markdown(r"""Here $x = n, m, h$ and $\alpha$ and $\beta$ are the opening and closing rates for the voltage dependent ion channel""")

        st.markdown(""" The mathematical form for the voltage-dependent opening and closing rates of ion channels was determined using the voltage-clamp technique. This method 
                    holds the membrane potential constant at a specific value while measuring the resulting ionic current. By repeating this process across different membrane 
                    potentials and analyzing the resulting current curves, Hodgkin and Huxley could determine the voltage-dependent kinetics of channel opening and closing.""")
        
        st.markdown("""The temperature at which the experiments performed by Hodgkin and Huxley was around 6 - 7$^{o}C$. As seen in the section of temperature dependence, 
                    the incorporation of biophysical details in the Hodgkin Huxley model allows for exploring the model dynamics at various temperatures and analyzing how various aspects
                    of action potential generation change upon temperature. One such aspect is the excitability and the action potential firing rate under the injection of a current pulse. 
                    In the interactive tool below, you can explore these dynamics. First, change the injected current while keeping other parameters constant and observe how the action 
                    potential changes. You can also modify other parameters like Q10 values or temperature and then vary the current to see their effects.""")
        try:
            image = Image.open("../streamlit_images/HH.png")
            st.image(image, caption="HH model", use_container_width=True)
        except FileNotFoundError:
            st.info("Image not found")

def display_hvcra_theory():
    """
    Display the theoretical background of the HVC(RA) model.
    """
    with st.expander('About HVC(RA) model'):
        st.markdown("""
        ## About HVC(RA) model

        A complex dynamical model that mimics the biophysical mechanisms involved in action potential generation
        for an excitatory class of neurons in the premotor brain region HVC in songbirds. This model involves the
        complex morphology of the HVC(RA) neurons broken down into two compartments, a dendrite and a soma which
        are coupled via ohmic coupling. The dendritic compartment contains a Calcium ion channel, which imparts 
        the model with eliciting a wide calcium spike. This in turn drives 4-5 tightly chunked sodium ion spikes
        in the somatic compartment. This behavior of the model is representative of the bursting behavior observed 
        in HVC(RA) neurons.

        ### Ion channels incorporated
            Dendrite:
                - Calcium channel
                - Calcium concentration dependent Potassium channel
                - Leak channel
                    
            Soma:
                - Sodium channel
                - Potassium channel
                - Leak channel
        """)

def display_hvci_theory():
    """
    Display the theoretical background for HVCI.
    """
    with st.expander('About HVC(I)) model'):
        st.markdown("""
        ## About HVC(I) model

        A complex dynamical model that mimics the biophysical mechanisms involved in action potential generation
        for an inhibitory class of neurons in the premotor brain region HVC in songbirds. This model involves the
        complex morphology of the HVC(I) neuron broken down to a single somatic compartment, which exhibits fast-spiking
        behavior, representative of the behavior observed in HVC(I) neurons.

        ### Ion channels incorporated
                - Sodium channel
                - Delay rectified Potassium channel
                - Fast Potassium channel
                - Leak channel
        """)

def create_current_stimulus_array(time_array, i_amp, i_start, i_end):
    '''
    Function for creating a pulse current input
    Args:
        time_array (numpy.ndarray): The time points at which to evaluate the current stimulus.
        i_amp (float): The amplitude of the current pulse.
        i_start (float): The start time of the current pulse.
        i_end (float): The end time of the current pulse.
    Returns:
        numpy.ndarray: A numpy array containing current input values at different time points.
    '''
    current_stimulus = np.zeros_like(time_array)
    start_id = np.argmin(np.abs(time_array - i_start))
    end_id = np.argmin(np.abs(time_array - i_end))
    current_stimulus[start_id:end_id] = i_amp
    return current_stimulus

def create_synapse_stimulus_array(time_array, temp, q_gate, g_max, g_start, step_size):
    '''
    Function for creating a single kick and decay type synaptic input
    Args:
        time_array (numpy.ndarray): The time points at which to evaluate the synaptic stimulus.
        temp (float): The temperature in Celsius.
        q_gate (float): The Q10 value for conformation dependent processes.
        g_max (float): The maximum synaptic conductance strength.
        g_start (float): The start time of the synaptic input.
        step_size (float): The time step size for the simulation.
    Returns:

        numpy.ndarray: A numpy array containing values of the synaptic conductance strength at different time points.
    '''
    synapse_stimulus = np.zeros_like(time_array)
    start_id = np.argmin(np.abs(time_array - g_start))
    q = pow(q_gate, 0.1 * (40.0 - temp))
    for i in range(len(time_array) - start_id):
        synapse_stimulus[start_id + i] = g_max * np.exp(-i * step_size / ( 2 * q ))

    return synapse_stimulus

def generate_poisson_events(event_time, freq):

    rand_num = np.random.uniform(0.0, 1.0)
    event_time = event_time - (log(rand_num) * 1000 / freq)
   
    return event_time

def create_synapse_stimulus_over_intervals_array(time_array, temp, q_gate, g_max, freq, interval_size, step_size, simulation_time):
    '''
    Function for creating synaptic inputs, that exist over a number of intervals in time
    Args:
        time_array (numpy.ndarray): The time points at which to evaluate the synaptic stimulus.
        temp (float): The temperature in Celsius.
        q_gate (float): The Q10 value for conformation dependent processes.
        g_max (float): The maximum synaptic conductance strength.
        freq (float): Frequency with which the inputs are made.
        interval_size (float): size of the time intervals over which the input is made
        step_size (float): The time step size for the simulation
        simulation_time (float): Time for which simulation is run
    Returns:

        numpy.ndarray: A numpy array containing values of the synaptic conductance strength at different time points.
    '''

    synapse_stimulus = np.zeros_like(time_array)
    q = pow(q_gate, 0.1 * (40.0 - temp))

    num_inputs = 5
    start_id = [np.argmin(np.abs(time_array - 150.0 * (i+1))) for i in range(num_inputs)]
    interval_steps = int(interval_size / step_size)
    end_id = [start_id[i] + interval_steps for i in range(num_inputs)]

    event_time = 0.0
    np.random.seed(1995)
    num_steps = int(simulation_time / step_size)

    for i in range(num_steps):
        
        event_count = 0

        make_input = any(start <= i < end for start, end in zip(start_id, end_id))
        if make_input:
            while True:
                if (i + 0.5) * step_size <= event_time < (i + 1.5) * step_size:
                    event_count += 1
                    event_time = generate_poisson_events(event_time, freq)
                elif event_time < (i + 0.5) * step_size:
                    event_time = generate_poisson_events(event_time, freq)
                else:
                    break
        
        kick_strength = 0
        if event_count == 0:
            if synapse_stimulus[i-1] != 0:
                synapse_stimulus[i] = synapse_stimulus[i-1] * exp(-step_size / (2 * q))
        else:
            for _ in range(event_count):
                kick_strength += np.random.uniform(0.0, 1.0) * g_max

            synapse_stimulus[i] = synapse_stimulus[i-1] + kick_strength
            event_count = 0

    return synapse_stimulus

def response_time(time, vd, vs, current_amplitude=None, current_input_start_time=None, excitatory_input_start_time=None, excitatory_input_strength=None, threshold=-20):
    """
    Assess whether neuron spiked and return a response time for the soma
    The response time is defined as the time for somatic membrane potential to reach a peak value from input time for subthreshold response 
    or the time to reach -20 mV for superthreshold response

    Args:
        time (numpy.ndarray): The time points at which to evaluate the response.
        vd (numpy.ndarray): The dendritic membrane potential over time.
        vs (numpy.ndarray): The somatic membrane potential over time.
        current_amplitude (float, optional): The amplitude of the current stimulus.
        current_input_start_time (float, optional): The start time of the current stimulus.
        excitatory_input_start_time (float, optional): The start time of the excitatory input.
        excitatory_input_strength (float, optional): The strength of the excitatory input.
        threshold (float, optional): The voltage threshold for spike detection.

    Returns:
        float: The response time for the soma.

    """
    time = time.tolist()
    vs = vs.tolist()
    vd = vd.tolist()

    if current_amplitude is not None:
        if current_amplitude > 0:

            if max(vd) > threshold and max(vs) > threshold:
                response_time = np.round(time[next(x[0] for x in enumerate(vs) if x[1] > threshold)] - current_input_start_time, 3)
                return response_time
            elif max(vd) < threshold and max(vs) > threshold:
                response_time = np.round(time[next(x[0] for x in enumerate(vs) if x[1] > threshold)] - current_input_start_time, 3)
                return response_time
            elif max(vd) < threshold and max(vs) < threshold:
                response_time = np.round(time[vs.index(max(vs))] - current_input_start_time, 3)
                return response_time
    
    elif excitatory_input_strength is not None:
        if excitatory_input_strength > 0:
            if max(vd) > threshold and max(vs) > threshold:
                response_time = np.round(time[next(x[0] for x in enumerate(vs) if x[1] > threshold)] - excitatory_input_start_time, 3)
                return response_time
            elif max(vd) < threshold and max(vs) > threshold:
                response_time = np.round(time[next(x[0] for x in enumerate(vs) if x[1] > threshold)] - excitatory_input_start_time, 3)
                return response_time
            elif max(vd) < threshold and max(vs) < threshold:
                response_time = np.round(time[vs.index(max(vs))] - excitatory_input_start_time, 3)
                return response_time
            
def create_sidebar_controls_lif():
    '''
    Sidebar controls for the Leaky Integrate and Fire neuron model

    Returns:
    -------
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
    """
    Prepare the plots for the Leaky Integrate and Fire neuron model
    """
    params = create_sidebar_controls_lif()

    neuron = LIF()

    STEP_SIZE = 0.01  # ms
    SIMULATION_TIME = 200.0 # ms
    time = np.arange(0, SIMULATION_TIME, STEP_SIZE)

    i_amp = params['I_amp']
    i_start = 20.0
    i_end = SIMULATION_TIME - 20.0

    lif_current_list = st.session_state.lif_current_list
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
            
    elif current_exists and current_changed:
        st.info(f"Current {i_amp:.1f} $\mu A/cm^2$ already tested")
    
    st.session_state.lif_last_current = i_amp
    
    return v, time, current_stimulus, st.session_state.lif_current_list, st.session_state.lif_frequency_list, st.session_state.lif_last_current

def create_sidebar_controls_hh():
    '''
    Sidebar controls for the Hodgkin Huxley (HH) neuron model

    Returns:
    -------
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
    
    def handle_reset():
        st.session_state.hh_current_list = []
        st.session_state.hh_frequency_list_control = []
        st.session_state.hh_frequency_list_alt = []
        st.session_state.hh_last_current = 0.0
        st.session_state.hh_reset_counter += 1

    hh_reset_key = st.session_state.hh_reset_counter

    st.sidebar.header('Model Parameters')
    st.sidebar.subheader('Temperature and Q10 values')
    q_gate = st.sidebar.slider('Q10 for conformation dependent processes', 2.0, 4.0, 3.0, 0.1, on_change=handle_reset)
    q_cond = st.sidebar.slider('Q10 for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1, on_change=handle_reset)

    st.sidebar.subheader('Current stimulus settings')
    i_amp = st.sidebar.slider('Current amplitude ($\mu A/cm^2$)', -1.0, 15.0, 0.0, 1.0, key=f'i_amp_{hh_reset_key}')
    
    temperature = st.selectbox('Temperature in Celsius (control fixed at 6.3 Celsius)', [0.0, 10.0, -10.0], width=300, on_change=handle_reset) # control temperature = 6.3 celsius

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
    """
    Prepare the plots for the Hodgkin Huxley neuron model
    """
    
    params = create_sidebar_controls_hh()
    temperature = params['temperature']
    q_gate = params['Q_gate']
    q_cond = params['Q_cond']

    neuron_control = HodgkinHuxley(6.3, q_gate, q_cond)
    neuron_alt = HodgkinHuxley(temperature, q_gate, q_cond)

    STEP_SIZE = 0.01  # ms
    SIMULATION_TIME = 300.0 # ms
    time = np.arange(0, SIMULATION_TIME, STEP_SIZE)

    i_amp = params['I_amp']
    i_start = 50.0
    i_end = SIMULATION_TIME - 50.0

    current_stimulus = create_current_stimulus_array(time,
                                                    i_amp,
                                                    i_start,
                                                    i_end
                                                    )
    
    solution_control = neuron_control.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
    
    solution_alt = neuron_alt.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)

    v = solution_control[:, 0]
    n = solution_control[:, 1]
    m = solution_control[:, 2]
    h = solution_control[:, 3]
    v_ = solution_alt[:, 0]
    n_ = solution_alt[:, 1]
    m_ = solution_alt[:, 2]
    h_ = solution_alt[:, 3]

    hh_current_list = st.session_state.hh_current_list
    hh_last_current = st.session_state.hh_last_current

    current_changed = abs(i_amp - hh_last_current) >= 0.5
    current_exists = any(abs(c - i_amp) == 0.00 for c in hh_current_list)    

    if current_changed and not current_exists:

        with st.spinner(f"Running F-I simulation for {i_amp:.1f} $\mu A/cm^{2}$..."):
            spike_indices = []
            spike_indices_ = []
            threshold = -20
            
            solution_control = neuron_control.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
            solution_alt = neuron_alt.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
        
            v = solution_control[:, 0]
            n = solution_control[:, 1]
            m = solution_control[:, 2]
            h = solution_control[:, 3]
            v_ = solution_alt[:, 0]
            n_ = solution_alt[:, 1]
            m_ = solution_alt[:, 2]
            h_ = solution_alt[:, 3]

            st.session_state.hh_current_list.append(i_amp)

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

            if len(spike_indices_) > 0 and not any(c < 0 for c in st.session_state.hh_current_list):
                duration_ = i_end - i_start
                frequency_alt = 1000 * len(spike_indices_) / duration_
            else:
                frequency_alt = 0

            
            st.session_state.hh_frequency_list_control.append(frequency_control)
            st.session_state.hh_frequency_list_alt.append(frequency_alt)

            sorted_pairs = sorted(zip(st.session_state.hh_current_list, st.session_state.hh_frequency_list_control, st.session_state.hh_frequency_list_alt))
            st.session_state.hh_current_list, st.session_state.hh_frequency_list_control, st.session_state.hh_frequency_list_alt = zip(*sorted_pairs)
            st.session_state.hh_current_list = list(st.session_state.hh_current_list)
            st.session_state.hh_frequency_list_control = list(st.session_state.hh_frequency_list_control)
            st.session_state.hh_frequency_list_alt = list(st.session_state.hh_frequency_list_alt)
            
            st.success(f"Added: {i_amp:.1f} $\mu A/cm^{2}$")
    
    elif current_exists and current_changed:
        st.info(f"Current {i_amp:.1f} $\mu A/cm^2$ already tested")

    st.session_state.hh_last_current = i_amp    

    return v, n, m, h, v_, n_, m_, h_, time, current_stimulus, temperature, st.session_state.hh_current_list, st.session_state.hh_frequency_list_control, st.session_state.hh_frequency_list_alt, st.session_state.hh_last_current

def create_sidebar_controls_hvcra():
    '''
    Sidebar controls for the HVC(RA) projection neuron model

    Returns:
    -------
    params : dict
    Dictionary containing all parameter values for the HVC(RA)
    '''

    st.sidebar.header('Model Parameters')
    st.sidebar.subheader('Temperature and Q10 values')

    
    input_type = st.selectbox('Input type', ['Current input', 'Synaptic input'], width=200)

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
        
        def handle_reset():
            st.session_state.hvcra_current_input_list = []
            st.session_state.hvcra_frequency_control_list = []
            st.session_state.hvcra_frequency_alt_list = []
            st.session_state.hvcra_last_current_input = 0.0
            st.session_state.hvcra_reset_counter += 1

        hvcra_reset_key = st.session_state.hvcra_reset_counter

        temperature = st.selectbox('Altered Temperature in $^o$C (control set to 40$^o$ C)', [30.0, 35.0], width=300, on_change=handle_reset) #  control temperature = 40.0 celsius
        
        q_gate = st.sidebar.slider('Q10 for conformation dependent processes', 2.0, 4.0, 3.0, 0.1, on_change=handle_reset)
        q_cond = st.sidebar.slider('Q10 for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1, on_change=handle_reset)

        st.sidebar.subheader('Current stimulus settings')

        i_amp = st.sidebar.slider(f'Current amplitude ($\mu A/cm^{2}$)', -0.2, 1.0, 0.00, 0.1, key=f'i_amp_{hvcra_reset_key}')

        return {
            'temperature': temperature,
            'Q_gate': q_gate,
            'Q_cond': q_cond,
            'I_amp': i_amp,
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

        def handle_reset():
            st.session_state.hvcra_synaptic_input_list = []
            st.session_state.response_time_control_list = []
            st.session_state.response_time_alt_list = []
            st.session_state.hvcra_last_synaptic_input = 0.0
            st.session_state.hvcra_reset_counter += 1

        hvcra_reset_key = st.session_state.hvcra_reset_counter

        temperature = st.selectbox('Altered Temperature in $^o$C (control set to 40$^o$ C)', [30.0, 35.0], width=300, on_change=handle_reset) #  control temperature = 40.0 celsius
        
        q_gate = st.sidebar.slider('Q10 for conformation dependent processes', 2.0, 4.0, 3.0, 0.1, on_change=handle_reset)
        q_cond = st.sidebar.slider('Q10 for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1, on_change=handle_reset)

        external_input = 'No'
        noise_input = 'No'
        '''
        col1, col2 = st.columns(2)
        with col1:
            external_input = st.selectbox('Add External Input', ['No', 'Yes'], width=200)
        with col2:
            noise_input = st.selectbox('Add Noise Input', ['No', 'Yes'], width=200)
        '''
        st.sidebar.subheader('Synaptic Input Settings')

        ge_max = st.sidebar.slider('Excitatory Synapse Strength (mS/cm^2)', 0.01, 1.0, 0.01, 0.01, key=f'ge_max_{hvcra_reset_key}')

        gi_max = 0.0 #st.sidebar.slider('inhibitory synapse strength (mS/cm^2)', 0.0, 0.5, 0.0, 0.05, key=f'gi_max_{hvcra_reset_key}')
        gi_start = 0.0 #st.sidebar.slider('inhibitory synapse start time (ms)', 100.0, 200.0, 150.0, 0.5, key=f'gi_start_{hvcra_reset_key}')

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

    #input_type = st.selectbox('Input type', ['Current input', 'Noise input'])
    input_type = st.selectbox('Input type', ['Single current pulse', 'Synaptic input in multiple intervals'])

    st.sidebar.subheader('Temperature and Q10 values')
    
    if input_type == 'Single current pulse':

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

        def handle_reset():
            st.session_state.hvci_current_input_list = []
            st.session_state.hvci_frequency_control_list = []
            st.session_state.hvci_frequency_alt_list = []
            st.session_state.hvci_last_current_input = 0.0
            st.session_state.hvci_reset_counter += 1 

        temperature = st.selectbox('Temperature in Celsius', [30.0, 35.0], on_change=handle_reset) #  control temperature = 40.0 celsius
        q_gate = st.sidebar.slider('Q10 for conformation dependent processes', 2.0, 4.0, 3.0, 0.1, on_change=handle_reset)
        q_cond = st.sidebar.slider('Q10 for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1, on_change=handle_reset)

        st.sidebar.subheader('Current stimulus settings')

        i_amp = st.sidebar.slider(f'Current amplitude ($\mu A/cm^{2}$)', 0.0, 20.0, 0.0, 2.5, key=f'i_amp_{hvci_reset_key}')
        
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
    
    if input_type == 'Synaptic input in multiple intervals':

        if 'hvci_synaptic_input_list' not in st.session_state:
            st.session_state.hvci_synaptic_input_list = []
        if 'hvci_frequency_control_list' not in st.session_state:
            st.session_state.hvci_frequency_control_list = []
        if 'hvci_frequency_alt_list' not in st.session_state:
            st.session_state.hvci_frequency_alt_list = []
        if 'hvci_last_synaptic_input' not in st.session_state:
            st.session_state.hvci_last_synaptic_input = 0.0
        
        if 'hvci_reset_counter' not in st.session_state:
            st.session_state.hvci_reset_counter = 0
        
        hvci_reset_key = st.session_state.hvci_reset_counter

        def handle_reset():
            st.session_state.hvci_synaptic_input_list = []
            st.session_state.hvci_frequency_control_list = []
            st.session_state.hvci_frequency_alt_list = []
            st.session_state.hvci_last_synaptic_input = 0.0
            st.session_state.hvci_reset_counter += 1 

        temperature = st.selectbox('Temperature in Celsius', [30.0, 35.0], on_change=handle_reset) #  control temperature = 40.0 celsius
        q_gate = st.sidebar.slider('Q10 for conformation dependent processes', 2.0, 4.0, 3.0, 0.1, on_change=handle_reset)
        q_cond = st.sidebar.slider('Q10 for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1, on_change=handle_reset)

        st.sidebar.subheader('Synaptic input settings')

        freq_input = st.sidebar.slider('input frequency (Hz)', 50.0, 300.0, 200.0, 50.0, on_change=handle_reset)
        g_max = st.sidebar.slider(f'input max kick ($mS/cm^{2}$)', 0.0, 5.0, 0.0, 0.5, key=f'i_amp_{hvci_reset_key}')
        interval_size = st.sidebar.slider(f'size of input intervals (ms)', 20.0, 60.0, 20.0, 10.0, on_change=handle_reset)
        
        return {
            'temperature': temperature,
            'Q_gate': q_gate,
            'Q_cond': q_cond,
            'Input_type': input_type,
            'synaptic_input_list': st.session_state.hvci_synaptic_input_list,
            'Frequency_control_list': st.session_state.hvci_frequency_control_list,
            'Frequency_alt_list': st.session_state.hvci_frequency_alt_list,
            'last_input': st.session_state.hvci_last_synaptic_input,
            'freq_input': freq_input,
            'input_kick': g_max,
            'interval_size': interval_size
        }

def prepare_hvcra_plots():
    """
    Prepare plots for the HVC(RA) model simulations.
    """

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

    response_time_control = None
    response_time_alt = None

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
        hvcra_last_current = st.session_state.hvcra_last_current_input

        current_changed = abs(i_amp - hvcra_last_current) > 0.01
        current_exists = any(c == i_amp for c in hvcra_current_input_list)

        if current_changed and not current_exists:

            with st.spinner(f"Running simulation for {i_amp:.2f} $\mu A/cm^{2}$..."):
                
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

                st.success(f"finished running for current input {i_amp:.2f} $\mu A/cm^{2}$!")

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
            st.info(f"Current input {i_amp:.2f} $\mu A/cm^{2}$already tested")
        
        st.session_state.hvcra_last_current_input = i_amp

        return input_type, q_cond, fluctuations, vs, vs_, vd, vd_, time, current_stimulus, current_stimulus, temperature, st.session_state.hvcra_current_input_list, st.session_state.hvcra_frequency_control_list, st.session_state.hvcra_frequency_alt_list, st.session_state.hvcra_last_current_input
 
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
        hvcra_last_synaptic_input = st.session_state.hvcra_last_synaptic_input 

        input_changed = abs(ge_max - hvcra_last_synaptic_input) > 0.005
        input_exists = any(g == ge_max for g in hvcra_synaptic_input_list)

        if input_changed and not input_exists:

            with st.spinner(f"Running simulation for {ge_max:.2f} $mS/cm^{2}$..."):

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

                response_time_control = response_time(time, 
                                                        vd, 
                                                        vs, 
                                                        excitatory_input_start_time=ge_start, 
                                                        excitatory_input_strength=ge_max)
                
                response_time_alt = response_time(time, 
                                                        vd_, 
                                                        vs_, 
                                                        excitatory_input_start_time=ge_start, 
                                                        excitatory_input_strength=ge_max)
                
                st.session_state.hvcra_synaptic_input_list.append(ge_max)
                st.session_state.response_time_control_list.append(response_time_control)
                st.session_state.response_time_alt_list.append(response_time_alt)
                st.session_state.hvcra_last_synaptic_input = ge_max

                sorted_pairs = sorted(zip(st.session_state.hvcra_synaptic_input_list, st.session_state.response_time_control_list, st.session_state.response_time_alt_list))
                st.session_state.hvcra_synaptic_input_list, st.session_state.response_time_control_list, st.session_state.response_time_alt_list = zip(*sorted_pairs)
                st.session_state.hvcra_synaptic_input_list = list(st.session_state.hvcra_synaptic_input_list)
                st.session_state.response_time_control_list = list(st.session_state.response_time_control_list)
                st.session_state.response_time_alt_list = list(st.session_state.response_time_alt_list)


                st.success(f"finished running for synaptic input {ge_max:.2f} $mS/cm^{2}$!")
            
        elif input_exists and input_changed:
            st.info(f"Synaptic input {ge_max:.2f} $mS/cm^{2}$ already tested")
            st.session_state.hvcra_last_synaptic_input = ge_max
            
        return input_type, q_cond, fluctuations, vs, vs_, vd, vd_, time, excitatory_synapse_stimulus_control, excitatory_synapse_stimulus_alt, temperature, st.session_state.hvcra_synaptic_input_list, st.session_state.response_time_control_list, st.session_state.response_time_alt_list, st.session_state.hvcra_last_synaptic_input

def prepare_hvci_plots():
    """
    Prepare plots for the HVC(I) model.
    """

    params = create_sidebar_controls_hvci()
    temperature = params['temperature']
    q_gate = params['Q_gate']
    q_cond = params['Q_cond']

    neuron_control = HvcInterNeuron(40.0, q_gate, q_cond)
    neuron_alt = HvcInterNeuron(temperature, q_gate, q_cond)

    input_type = params['Input_type']

    STEP_SIZE = 0.01  # ms
    if input_type == 'Single current pulse':
        simulation_time = 300.0 # ms
    else:
        simulation_time = 1000.0
    time = np.arange(0, simulation_time, STEP_SIZE)

    v = np.full((10000, 1), 0)
    v_ = np.full((10000, 1), 0)

    solution_alt = None
    solution_control = None
    current_stimulus = None
    freq_noise = None
    noise_strength = None
    inhibitory_synapse_stimulus_control = None
    inhibitory_synapse_stimulus_alt = None

    if input_type == 'Single current pulse':
    
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
        hvci_last_current = st.session_state.hvci_last_current_input

        current_changed = abs(i_amp - hvci_last_current) > 0.01
        current_exists = any(c == i_amp for c in hvci_current_input_list)

        if current_changed and not current_exists:

            with st.spinner(f"Running simulation for {i_amp:.2f} $\mu A/cm^{2}$..."):

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

                st.success(f"finished running for {i_amp:.2f} $\mu A/cm^{2}$!")

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
                    
                frequency_control = spike_count_control * 1000 / (i_end - i_start)
                frequency_alt = spike_count_alt * 1000 / (i_end - i_start)

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
            st.info(f"Current input {i_amp:.2f} $\mu A/cm^{2}$ already tested")
        
        st.session_state.hvci_last_current_input = i_amp
        
        return input_type, v, v_, time, current_stimulus, temperature, st.session_state.hvci_current_input_list, st.session_state.hvci_frequency_control_list, st.session_state.hvci_frequency_alt_list, st.session_state.hvci_last_current_input

    if input_type == 'Synaptic input in multiple intervals':
        
        input_strength = params['input_kick']
        freq_input = params['freq_input']
        interval_size = params['interval_size'] 
        
        excitatory_synapse_stimulus_control = create_synapse_stimulus_over_intervals_array(time,
                                                40.0,
                                                q_gate,
                                                input_strength,
                                                freq_input,
                                                interval_size,
                                                STEP_SIZE,
                                                simulation_time
                                                )
        
        excitatory_synapse_stimulus_alt = create_synapse_stimulus_over_intervals_array(time,
                                                temperature,
                                                q_gate,
                                                input_strength,
                                                freq_input,
                                                interval_size,
                                                STEP_SIZE,
                                                simulation_time
                                                )
        
        inhibitory_synapse_stimulus_control = create_synapse_stimulus_array(time,
                                                40.0,
                                                q_gate,
                                                0,
                                                0,
                                                STEP_SIZE
                                                )
        
        inhibitory_synapse_stimulus_alt = create_synapse_stimulus_array(time,
                                                temperature,
                                                q_gate,
                                                0,
                                                0,
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
        
        hvci_input_list = st.session_state.hvci_synaptic_input_list
        hvci_last_input = st.session_state.hvci_last_synaptic_input

        input_changed = abs(input_strength - hvci_last_input) > 0.01
        input_exists = any(n == input_strength for n in hvci_input_list)

        if input_changed and not input_exists:

            with st.spinner(f"Running simulation for {input_strength:.2f} $mS/cm^{2}$..."):

                excitatory_synapse_stimulus_control = create_synapse_stimulus_over_intervals_array(time,
                                                        40.0,
                                                        q_gate,
                                                        input_strength,
                                                        freq_input,
                                                        interval_size,
                                                        STEP_SIZE,
                                                        simulation_time
                                                        )
                
                excitatory_synapse_stimulus_alt = create_synapse_stimulus_over_intervals_array(time,
                                                        temperature,
                                                        q_gate,
                                                        input_strength,
                                                        freq_input,
                                                        interval_size,
                                                        STEP_SIZE,
                                                        simulation_time
                                                        )
                
                inhibitory_synapse_stimulus_control = create_synapse_stimulus_array(time,
                                                                                    40.0,
                                                                                    q_gate,
                                                                                    0,
                                                                                    0,
                                                                                    STEP_SIZE
                                                                                    )
                
                inhibitory_synapse_stimulus_alt = create_synapse_stimulus_array(time,
                                                                                temperature,
                                                                                q_gate,
                                                                                0,
                                                                                0,
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

                spike_count_control = 0
                spike_count_alt = 0
                threshold = -20           
        
                for i in range(1, len(v)):
                    if v[i-1] < threshold and v[i] >= threshold:
                        spike_count_control += 1
                    if v_[i-1] < threshold and v_[i] >= threshold:
                        spike_count_alt += 1
                    
                frequency_control = spike_count_control * 1000 / simulation_time
                frequency_alt = spike_count_alt * 1000 / simulation_time

                st.session_state.hvci_synaptic_input_list.append(input_strength)
                st.session_state.hvci_frequency_control_list.append(frequency_control)
                st.session_state.hvci_frequency_alt_list.append(frequency_alt)
                st.session_state.hvci_last_synaptic_input = input_strength

                sorted_pairs = sorted(zip(st.session_state.hvci_synaptic_input_list, st.session_state.hvci_frequency_control_list, st.session_state.hvci_frequency_alt_list))
                st.session_state.hvci_synaptic_input_list, st.session_state.hvci_frequency_control_list, st.session_state.hvci_frequency_alt_list = zip(*sorted_pairs)
                st.session_state.hvci_synaptic_input_list = list(st.session_state.hvci_synaptic_input_list)
                st.session_state.hvci_frequency_control_list = list(st.session_state.hvci_frequency_control_list)
                st.session_state.hvci_frequency_alt_list = list(st.session_state.hvci_frequency_alt_list)
            
        elif input_exists and input_changed:
            st.info(f"Synaptic input {input_strength:.2f} $mS/cm^{2}$ already tested")
        
        st.session_state.hvci_last_synaptic_input = input_strength
        
        return input_type, v, v_, time, excitatory_synapse_stimulus_control, temperature, st.session_state.hvci_synaptic_input_list, st.session_state.hvci_frequency_control_list, st.session_state.hvci_frequency_alt_list, st.session_state.hvci_last_synaptic_input


