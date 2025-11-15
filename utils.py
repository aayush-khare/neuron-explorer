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
    """
    Display text for the introduction page.
    """
    
    st.markdown('<p class="big-font"> Neurons are excitable cells that generate electrical signals known as action potentials or simply spikes. ' \
                'A spike in a neuron triggers release of chemical messengers known as neurotransmitters. Depending on the type of neurotransmitter ' \
                'released (which in turn depends on the neuron type), it can in turn stimulate or inhibit spike generation in other neurons. ' \
                'It is through these electrical and chemical processes that neurons communicate with each other and the resulting pattern of activity ' \
                'encodes for information in the brain. ' \
                'Neurons are the fundamental units of our nervous system. To understand how this system functions, particularly our brain, we must first ' \
                'understand how these cells receive, process and transmit information. </p>', unsafe_allow_html=True) \
    
    neuron_image = os.path.join(image_dir, "neuron.jpg")
    st.image(neuron_image, width=1000, caption='Credits: Corbis/Getty Images')
        
    st.markdown('<p class="big-font"> In this interactive tool, you get to explore some computational models that are used for simulating the dynamics of a neuron cell. \
                 You will start by learning about the parts of a neuron, and the function they serve. You will also learn about the electrical properties and how Physics concepts can ' \
                 'be utilized along with biological details to mathematically model neural dynamics and simulate them on a computer. ' \
                 'You will also learn how the various processes behind neural function depend on temperature, and how that affects properties associated with ' \
                 'neural activity. </p>', unsafe_allow_html=True)

def display_electrical_properties():
    """
    Text and images for the page discussing electrical properties
    and temperature dependence of neural dynamics.
    """

    st.markdown("### Parts of a neuron")

    image_folder_0 = os.path.join(image_dir, "neuron_parts")
    image_files_0 = sorted([f for f in os.listdir(image_folder_0) if f.endswith(('.png'))])

    descriptions_0 = [
    "A Neuron cell has a complex morphology with 3 main parts, the dendrite, the soma and the axon.",
    "A neuron receives information or inputs from other neurons through the Dendrite. The Dendrite has extensive branches, often referred \n"
    "to as the Dendritic tree, which is where a neuron receives majority of its inputs from other neurons.",
    "The cell body of a neuron is called the Soma which contains the nucleus. The soma can also receive inputs from other neurons, and is the site of spike generation. ",
    "The spike is generated at the axon-hillock, the part of the soma that connects to the axon. ",
    "The spike is transmitted along the axon, a long cable like structure emerging from the soma. The ends of the axon, called the axon terminals are sites \n"
    "which contain the neurotransmitters. ",
    "A spike arriving at the terminals triggers the neurotransmitter release in the extracellular space. \n"
    "The neurotransmitters bind to receptors on the dendrites of other neurons, triggering electrical activity and continuing the flow of information."
    ]

    max_index_neuron = len(descriptions_0) - 1
    min_index_neuron = 0

    if 'img_index' not in st.session_state:
        st.session_state.img_index = 0
            
    col1, col2 = st.columns([0.5, 0.8])
    with col1:
        if st.session_state.img_index > min_index_neuron:
            if st.button("Prev "):
                st.session_state.img_index -= 1
                st.rerun()

    with col2:
        if st.session_state.img_index < max_index_neuron:
            if st.button("Next "):
                st.session_state.img_index += 1
                st.rerun()
    
    current_index = st.session_state.img_index
    current_image = Image.open(os.path.join(image_folder_0, image_files_0[current_index]))
    st.image(current_image, width=700)
    description_placeholder = st.empty()
    with description_placeholder.container():
        st.markdown(f"""
            <div style="min-height: 100px; padding: 10px;">
                {descriptions_0[current_index]}
            </div>
        """, unsafe_allow_html=True)
    #st.markdown(f"{descriptions_0[current_index]}")

######################################################################################################

    st.markdown("### Action potential generation in a neuron")
    image_folder_1 = os.path.join(image_dir, "neuron_dynamics")
    image_files_1 = sorted([f for f in os.listdir(image_folder_1) if f.endswith(('.png'))])

    descriptions_1 = [
    "There exists a difference in the potential on either side of the cell membrane. This difference, typically denoted by V "
    "will be referred to as the membrane potential, or simply the voltage of the neuron. In the resting state, this voltage is typically "
    "in the range of -65 to -80 mV. We use a simple picture above for the neuron, showing how the cellular membrane separates the intracellular "
    "and extracellular space.",

    "This potential difference exists because of a concentration difference of ions on either side of the membrane, such that the intracellular "
    "space is negatively charged relative to the extracellular. Additionally, specific ions are higher in concentration on one side "
    "of the membrane, as seen above for sodium and potassium. There are other ions as well, like Chloride and Calcium, which are not shown "
    "for simplicity. Later we will learn how the sodium and potassium ions particularly play an important role in action "
    "potential generation. "
    "The membrane is a lipid bilayer which is impermeable to the flow of ions. However, it contains various ion channels that selectively "
    "allow for the movement of specific ions. These ion channels can open and close through changes in the conformational states of the "
    "channel proteins. The opening and closing rates of these channels depend on the membrane potential, thus earning them the name "
    "voltage-gated ion channels. In the resting state, these voltage-gated channels are predominantly closed.",
    
    "A brief, excitatory input or injected current through an electrode can elevate the membrane potential from its "
    "resting state. This process is called membrane depolarization. As the membrane depolarizes, ion channels permeable to sodium ions "
    "open, allowing an influx of sodium ions that further aids depolarization.",
    
    "If the input were strong enough, the membrane potential can be depolarized enough to cross a value called the threshold potential, at which "
    "a large positive feedback loop emerges, as sodium ions surge into the cell resulting in rapid depolarization.",

    "The rapid depolarization subsequently opens voltage-gated potassium channels, allowing "
    "potassium ions to move out of the cell, which repolarizes the membrane back toward its resting potential.",

    "The rapid depolarization followed by repolarization, which typically occurs within a millisecond in most neurons, is known as "
    "an action potential. It is this sharp pulse of electrical activity, generated at the axon hillock, that is carried along the "
    "axon and triggers the release of neurotransmitters at the axon terminal. After an action potential, the concentrations of the Sodium and "
    "Potassium ions on either side of the membrane are re-established by active proteins called ion pumps, which requires energy that is "
    "achieved through the breakdown of ATP. The sodium-potassium pump moves 3 Sodium ions out of the cell for every 2 Potassium ions moved "
    "into the cell." 
    ]

    max_index_dynamics = len(descriptions_1) - 1
    min_index_dynamics = 0

    if 'img_index_1' not in st.session_state:
        st.session_state.img_index_1 = 0

    col1, col2 = st.columns([1, 0.75])
    with col1:
        if st.session_state.img_index_1 > min_index_dynamics:
            if st.button("Prev"):
                st.session_state.img_index_1 -= 1
                st.rerun()
                
    with col2:
        if st.session_state.img_index_1 < max_index_dynamics:
            if st.button("Next"): 
                st.session_state.img_index_1 += 1
                st.rerun()

    current_index_1 = st.session_state.img_index_1
    current_image_1 = Image.open(os.path.join(image_folder_1, image_files_1[current_index_1]))
    st.image(current_image_1, width=1000)
    
    description_placeholder = st.empty()
    with description_placeholder.container():
        st.markdown(f"""
            <div style="min-height: 200px; padding: 10px;">
                {descriptions_1[current_index_1]}
            </div>
        """, unsafe_allow_html=True)
    
######################################################################################################

    st.markdown("### Biophysics behind modeling the dynamics of a neuron")

    biophysics_image = os.path.join(image_dir, "biophysics.png")
    st.image(biophysics_image, width=800)
    st.markdown("" \
        "To simulate the dynamics of a neuron, we can model a small patch of the neuronal membrane as a capacitor due to the charge separation " \
        "across it. The key assumption is that this membrane patch is isopotential. This means that the voltage is uniform across the entire patch. " \
        "The charge separation and resulting voltage vary as ions flow in and out of the neuron. Not all ion channel conductances are constant (the arrow on " \
        "the resistor denotes variability). Instead, as we will see later, channels open and close following first-order kinetics with " \
        "voltage-dependent rates. In the resting state, when there are no external inputs the voltage remains constant because there is no net " \
        "ion flow. This can be mathematically written as the membrane potential equation: "
        )
    
    st.latex(r''' C\frac{dV}{dt} + I_{ion} = 0''')

    st.markdown("An ionic current, $I_{ion}$ at any instance is given as ")

    st.latex(r''' I_{ion} = \overline{g}_{ion}p^{x}(V - E_{rev, ion})''')

    st.markdown("$p$ denotes the probability of activation for a subunit constituting an ion channel, and the exponent denotes how many " \
    "such subunits must be independently activated or deactivated. $\\overline{g}_{ion}$ denotes the maximum conductance for a given ion channel. " \
    "Hence the driving force for an ionic current depends both on the state of the channel denoted by $p$ and the difference " \
    "$V - E_{rev, ion}$, not simply V. This additional term $E_{rev, ion}$, is the reversal potential. " \
    "It is the membrane potential value at which the direction of an ionic current reverses. " \
    "To understand this, consider the case when sodium ions flows into the cell. The influx depolarizes the cell, moving the potential towards " \
    "a positive value. At the same time due to the influx, the concentration of sodium ions increases in the intracellular space. " \
    "Hence, there is both a decrease in the attraction and an increase in repulsive forces, and at some value of the membrane potential, " \
    "the repulsive forces would balance out the attractive forces. This particular value is the reversal potential. " \
    "Positive ions like Sodium that flow into the cell have a positive reversal potential, while positive ions like Potassium " \
    "that flow out of the cell have a negative reversal potential.")

    st.markdown("In the presence of an injected current we have: ")

    st.latex(r''' C\frac{dV}{dt} + I_{ion} = I_{inj}''') 

    st.markdown("Taking the ionic current to the right hand side of the equation and noting the fact that the total ionic current would be the sum of the" \
    " contributions from the various specific ion channels, the membrane potential equation becomes ")

    st.latex(r''' C\frac{dV}{dt} = -\sum I_{ion} + I_{inj} ''')

    st.markdown("A positive injected current causes depolarization while a negative current causes hyperpolarization (makes the membrane " \
    "potential more negative with respect to the resting state).")

    st.markdown("### Synaptic inputs")

    st.markdown("In addition to an experimentally injected current, neurons can be depolarized or hyperpolarized due to synaptic inputs. These are " \
    "inputs from other neurons, and can be excitatory or inhibitory depending on the neurotransmitters released. When a neurotransmitter " \
    "binds to receptors on a neuron, it triggers a postsynaptic potential (PSP) which is a transient voltage change. PSPs can be either " \
    "excitatory (EPSP) or inhibitory (IPSP), and occur as a result of the influx or outflux of ions through ligand-gated ion channels. " \
    "These ion channels open as a result of the neurotransmitters binding to the receptors. Multiple EPSPs occurring in close temporal " \
    "proximity can add up and cause the voltage to cross the threshold required for action potential generation. In the presence of " \
    "synaptic inputs, the membrane potential equation becomes ")

    st.latex(r''' C\frac{dV}{dt} = -\sum I_{ion} + I_{inj} - I_{e, syn} - I_{i, syn}''')

    st.markdown("Synaptic currents follow the form $I_{syn} = g_{syn}(V - E_{rev, syn})$. The reversal potential for excitatory " \
    "synapses is typically taken to be 0 mV, while for inhibitory synapses is set to the resting voltage of the neuron. " \
    "Every time a neuron receives a synaptic input, the synaptic conductance undergoes a step increase $g \\longrightarrow g + G$ " \
    "by an amount corresponding to the strength ($\\textit{G}$) of the synaptic connection. In between synpatic inputs, the " \
    "synaptic conductance undergoes exponential decay given by ")

    st.latex(r''' \tau \frac{dg}{dt} = -g''')
    col1, col2, col3 = st.columns([4, 4, 1])
    with col2:
        synapse_image = os.path.join(image_dir, "synapse.png")
        st.image(synapse_image, width=200)

    st.markdown("The time constant for decays is set to 2 ms.")
#####################################################################################################

    st.markdown("### Temperature dependence of neural dynamics")

    st.markdown("Most biological and biochemical processes depend on the temperature of the environment in which they occur. " \
    "Depending on the underlying mechanisms, some processes are highly sensitive to temperature changes, while others are not. " \
    "The various processes that drive neural dynamics are no exception and are also temperature-dependent. This is a " \
    "particularly important consideration for electrophysiologists, who study action potential generation in neurons, since it " \
    "requires careful consideration of the temperature at which recordings are performed. " 
    "A common approach used involves preparing brain tissue slices and inserting electrodes into them to detect neural "
    "activity. In most cases, these experiments are carried out at room temperature. However, the normal brain temperature of "
    "the animal species from which the slices are taken may differ significantly from room temperature. This discrepancy "
    "necessitates proper estimation of how the neuronal dynamics and properties change with temperature for specific classes of"
    "neurons. It is an equally important consideration for computational modelers, who often aim to simulate brain dynamics under " \
    "physiological (normal) temperature conditions.")

    st.markdown("Temperature dependece for a biological process is quantified using a $Q_{10}$ value, which denotes the factor by which a rate scales" \
    "for a 10$^{o}$ C change in temperature. If a process occurs with a rate $r_{o}$ under " \
    "physiological conditions (Temperature $T_o$), then the rate at any other temperature $T$ is given as ")

    st.latex(r''' r(T) = r(T_o)Q_{10}^{\frac{T - T_o}{10}}''')

    st.markdown("The two main aspects involved in the action potential generation in neurons that we discussed above are the channel kinetics and the ion conductance. These two processes " \
    "play a crucial role in determining the action potential characteristics and are also temperature dependent.")

    image_folder_2 = os.path.join(image_dir, "temperature")
    image_files_2 = sorted([f for f in os.listdir(image_folder_2) if f.endswith(('.png'))])

    descriptions_2 = [
        "The movement of ions through the cellular medium is governed by a process called diffusion. This process affects the various conductances involved in the ion channel currents. "
        "The temperature dependence of this process is weak, quantified by a Q10 value typically in the range 1.2 - 1.4",
        "The conformational changes governing the opening and closing rates of ion channels are strongly affected by temperature changes. There temperature dependence can be quantified"
        "by Q10 values typically in the range 2 - 4."
    ]

    max_index_temperature = len(descriptions_2) - 1
    min_index_temperature = 0

    if 'img_index_2' not in st.session_state:
        st.session_state.img_index_2 = 0

    col1, col2 = st.columns([0.5, 0.6])
    with col1:
        if st.session_state.img_index_2 > min_index_temperature:
            if st.button("Prev  "):
                st.session_state.img_index_2 -= 1
                st.rerun()
                
    with col2:
        if st.session_state.img_index_2 < max_index_temperature:
            if st.button("Next  "):
                st.session_state.img_index_2 += 1
                st.rerun()
                
    current_index_2 = st.session_state.img_index_2
    current_image = Image.open(os.path.join(image_folder_2, image_files_2[current_index_2]))

    st.image(current_image, width=800)

    description_placeholder = st.empty()
    with description_placeholder.container():
        st.markdown(f"""
            <div style="min-height: 100px; padding: 10px;">
                {descriptions_2[current_index_2]}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("Reversal potentials are also directly proportional to the temperature (in Kelvin) $E_{rev} \\propto T_K$. Finally any time constants are "
    "inverse of activation and inactivation rates. Hence if rates have a $Q_{10}$ of 3, time constants would have a $Q_{10}$ = $\\frac{1}{3}$. " \
    "For a $Q_{10} = 1.3$ for conductances, when temperature decreases by 10$^{o}$ C, all conductances scale down by a factor of 1.3 " \
    "For a $Q_{10} = 3.0$ for rates, when temperature decreases by 10$^{o}$ C, all rates scale down by 3, while all time constants scale up by 3.")

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

    st.markdown(f"""

The Leaky Integrate and Fire (LIF) model is a simple mathematical model that effectively explains the most basic neuron 
function, i.e., integration of inputs and production of an output spike, without incorporating the complex mechanisms involved in the ion
channel dynamics we saw in the previous page. 

                
The Leaky in the LIF model corresponds to the incorporation of a leaky ion channel \
in the model, that ensures that the membrane potential exponentially decays back to the \
resting-state value in the absence of any inputs. This is the only ion channel incorporated \
in the model. The LIF model has an artificially defined threshold value. If a depolarizing input
causes the membrane potential to reach this threshold, the membrane potential is adjusted back to a reset 
value, and this is considered as a spike being generated by the model neuron. 

The membrane potential dynamics of the model are described by the equation """)
    
    st.latex(r''' C\frac{dV}{dt} = - I_L + I_{inj} ''')

    st.markdown(f"""Here the first term on RHS is the leak current given by""")

    st.latex(r''' I_L = g_L(V - E_L)''')

    st.markdown(f""" In the presence of a constant current input, the differential equation has an analytical solution for the time evolution of the membrane potential given by""")

    st.latex(r''' V(t) = E_L + I_{inj}R + (V(0) - E_L - I_{inj}R)e^{-\frac{t}{\tau}}''')

    st.markdown(f""" Here $R = \\frac{{{1}}}{{g_L}}$  and the time constant $\\tau = RC$. 
                This equation, holding for any value of the membrane potential below the threshold
                value, can be used to obtain a response time for the LIF model. This response time, say
                $t'$ corresponds to the time it takes for the neuron to fire an action potential from the
                time at which the current pulse starts. Setting the initial condition, $V(0) = E_L$ and $V(t') = V_{{th}}$, the threshold
                membrane potential, we get""")
    
    st.latex(r''' t' = \tau ln(\frac{I_{inj}R}{I_{inj}R + V(0) - V_{th}})''')

    st.markdown(f"""Rearranging the above equation """)

    st.latex(r'''t' = \tau ln(\frac{1}{1 + \frac{V(0) - V_{th}}{I_{inj}R}})''')

    st.markdown(f"""and using the approximation that $ln(1+x) \\approx x$ we get""")
    st.latex(r''' t' \approx C\frac{V_{th} - V(0)}{I_{inj}}''')

    st.markdown(f"""As can be seen, increasing the current would decrease the response time. If we choose a reset value to be equal to the 
                initial condition $V(0)$, then as long as the current pulse is present, the membrane potential depolarize to the 
                threshold in time t', followed by a reset and subsequent depolarization with the same time t', to the threshold. 
                 Hence the inverse of the above equation gives us the firing rate of the LIF model. It should 
                    be noted that the closer the initial value is to threshold membrane potential, faster the response. 
                And larger the current pulse, faster the response, and higher the firing rate. """)
    
    st.markdown(f""" Click on the Plots tab to explore the effect of an increasing injected current on the firing rate of the LIF neuron model.
                In this setup, we have defined the threshold to be -50 mV. Each time the membrane potential crosses the threshold, 
                a spike is generated and the membrane potential is reset to a value same as the leak reversal potential (-65 mV, which is also 
                the initial condition). If the current pulse is present for sufficient time, the neuron would depolarize to the threshold and 
                spike again.
                """)

def display_lif_summary():
    """
    Display concluding remarks for the LIF section.
    """

    with st.expander("Remarks"):
        st.write("f-I curves like above provide crucial information related to processing of inputs by neurons, like the minimum " \
        "current needed by a neuron to produce action potentials. While for the LIF model, it is evident that the firing rate keeps on " \
        "increasing with the input current pulse, there are some classes of neurons that exhibit saturation, which would show up as minimal " \
        "or no increase in firing rate after a certain current. Such aspects can be useful towards classifying neurons as well. ")
    
def display_hh_theory():
    """
    Display the theoretical background of the Hodgkin-Huxley model.
    """

    st.markdown("""
    
    Hodgkin Huxley model is a mathematical model that approximates the biophysical mechanisms involved in action potential generation. \
    Developed in 1952 by Alan Hodgkin and Andrew Huxley by studying the ionic mechanisms behind axon potential generation and propagation \
    in the squid giant axon, this model forms a basis for biophysical models describing neural dynamics for different classes of neurons, 
    and earned them the Nobel Prize in Physiology or Medicince.
    The HH model treats the cell membrane as a Capacitor, and voltage-dependent ion channels as variable resistors (variable as the resistance 
    or conductance at any instance depends on the membrane potential at that instant), that impart non-linearity to the model dynamics. \
    The mathematical form for the current flowing through the membrane is described as """)

    st.latex(r''' C\frac{dV}{dt} = - I_{Na} - I_{K} - I_L + I_{inj} ''')

    hh_image = os.path.join(image_dir, "hh/hh.png")
    col1, col2, col3 = st.columns([0.5, 1, 0.5])
    with col2:
        st.image(hh_image, width=600)

    st.markdown("""$I_{L}$ is the leak current and follows the same form as that in the LIF model. The sodium and potassium channel currents 
                are described by """)

    st.latex(r''' I_{Na} = \overline{g}_{Na}m^3h(V - V_{Na})''')

    st.latex(r''' I_K = \overline{g}_Kn^4(V - V_K)''')

    st.markdown(""" $\\overline{g}_{Na}$ and $\\overline{g}_K$ are fixed and are determined by the surface density of the respective channels 
                on the membrane. The variability in the ion channel conductances arises from the fraction of the channels that are either open 
                or closed. This is governed by the dynamics of the gating variables $n$, $m$ and $h$, which follow first order kinetics""")
    
    st.latex(r''' \frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)x''')

    st.markdown(r"""Here $x = n, m, h$ and $\alpha$ and $\beta$ are the opening and closing rates for the voltage dependent ion channel.""")

    st.markdown(""" The mathematical form for the voltage-dependent opening and closing rates of ion channels was determined using the voltage-clamp technique. This method 
                holds the membrane potential constant at a specific value while measuring the resulting ionic current. By repeating this process across different membrane 
                potentials and analyzing the resulting current curves, Hodgkin and Huxley could determine the voltage-dependent kinetics of channel opening ($\\alpha$)and closing ($\\beta$).
                The mathematical form for these kinetics typically is an exponential function or some combination of a linear and exponential function of the membrane potential. For the Hodgkin-Huxley
                model, these rates are as follows""")

    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'''\alpha_n(V) = \frac{0.01(V + 55)}{1 - e^{-\frac{V + 55}{10}}}   ''')
    with col2:
        st.latex(r'''\beta_n(V) = 0.125e^{-\frac{V+65}{60}}''')
    
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'''\alpha_m(V) = \frac{0.1(V+40)}{1 - e^{-\frac{V+40}{10}}}''')
    with col2:
        st.latex(r'''\beta_m(V) = 4e^{-\frac{V+65}{18}}''')
    
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'''\alpha_h(V) = 0.07e^{-\frac{V+65}{20}}''')
    with col2:
        st.latex(r'''\beta_h(V) = \frac{1}{1 + e^{-\frac{V+35}{10}}}''')
    
    st.markdown("""Hodgkin and Huxley performed their experiments at around 6 - 7$^{o}C$. As mentioned in the section of temperature dependence, 
                incorporation of biophysical details in the Hodgkin Huxley model allows us to explore the model dynamics at various temperatures 
                and analyze how various aspects of action potential generation change upon temperature. One such aspect is the excitability and 
                the action potential firing rate under the injection of a current pulse.  
                Navigate to the Plots tab to explore these dynamics. 
                First, change the injected current while keeping other parameters constant and observe how the neural response particularly the firing
                rate differs in the control condition (6.3$^{o}$ C) and an altered condition corresponding to a different temperature. You can 
                also modify other parameters like $Q_{10}$ values or temperature and then repeat the current change experiment to see the respective effects.""")

def display_hh_summary():
    """
    Display concluding remarks for the HH section.
    """

    with st.expander("Remarks"):
        st.write(" HH model provides a biophysical description of action potential generation by incorporating phenomenolgical variables " \
        "representing the opening and closing dynamics of the sodium and potassium ion channels. Combined with the knowledge about how " \
        "temperature affects various biological processes allows making predictions about change in neural dynamics with a change in temperature. " \
        "With the above tool, you learned how a drop in temperature raises the resting membrane potential due to its dependence on " \
        "the temperature dependent reversal potential of the leak channel. Furthermore a drop in temperature slows down the opening and closing " \
        "dynamics of ion channels (governed by the $Q_{10}$ of conformational change dependent processes). These dynamics govern " \
        "characteristic features of spikes, such as spike widths, and interspike intervals. Hence under the injection of a strong current pulse "
        "(fixed duration) at two different temperatures, the slowed dynamics reduce the action potential firing rate. ")
    
def display_hvc_background():
    """
    Display the background behind the role of brain area HVC in birdsong.
    """ 

    st.markdown("HVC (used as a proper name) is a brain area in the brain of songbirds that has been hypothesized to encode for features of birdsong " \
    "such as the duration of various song elements (syllables, silent gaps in between syllables), often referred to as song timing features. HVC is " \
    "analogous to the premotor cortex in human brain (associated with motor planning and execution).")

    descriptions = ["Zebra finches is a species of songbirds that is extensively studied to gain insights about the underlying neural circuitry and "
    "mechanisms that control behavior that is composed of a sequence of actions. The motivation arises from the simplicity of the song (composed of "
    "a fixed sequence of syllables) and the experimental tractability of probing the different brain regions.", 
    "HVC forms a part of the premotor pathway, projecting connections to downstream region Robust nucleus of the Arcopallium (RA), which in turn "
    "projects to the tracheosyringeal portion of the brainstem (nXIIts) that contains motoneurons controlling the vocal muscles. "]
    
    image_folder = os.path.join(image_dir, "zf")
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png'))])

    max_index = len(image_files) - 1
    min_index = 0

    if 'img_index_' not in st.session_state:
        st.session_state.img_index_ = 0

    col1, col2 = st.columns([1.2, 1])
    with col1:
        if st.session_state.img_index_ > min_index:
            if st.button("Prev  "):
                st.session_state.img_index_ -= 1
                st.rerun()
                
    with col2:
        if st.session_state.img_index_ < max_index:
            if st.button("Next  "):
                st.session_state.img_index_ += 1
                st.rerun()
                
    current_index_ = st.session_state.img_index_
    current_image_ = Image.open(os.path.join(image_folder, image_files[current_index_]))

    st.image(current_image_, width=800)
    st.markdown(f"{descriptions[current_index_]}")

def display_hvcra_theory():
    """
    Display the theoretical background of the HVC(RA) model.
    """
    st.markdown("""
    ## About HVC(RA) model

    This neuron model is a conductance-based compartment model, used to simulate the generation of a burst of action potentials
    for an excitatory class of neurons in the premotor brain region HVC (used as proper name) in songbirds. These neurons communicate 
    with neurons within the HVC, as well as send axonal projections to a downstream brain area called the Robust Nucleus of the 
    Arcopallium (RA) involved in song production. This particular connectivity is why these neurons are named HVC(RA) neurons.

    These neurons exhibit precisely-timed bursting behavior, locked with song auditory features. Each HVC(RA) neuron bursts exactly once during a
    rendition of the song motif. Furthermore, aligning these bursts with the song motif exhibits that the bursts occur in a sequence. It is this 
    observation which motivates studying birdsong to gain insights about how neural circuits could govern behavior that involves a sequence of actions or
    steps chunked together, like performing a dance move.  
                
    To simulate the dynamics of these neurons, the model the Hodgkin-Huxley formalism seen in the previous section, but now with the incorporation 
    of additional ion channels. Furthermore, the complex morphology of the HVC(RA) neurons is broken down into a dendritic 
    and somatic compartment coupled via ohmic coupling, to obtain a simple model that is able to exhibit precise bursting behavior. 
    The dendritic compartment contains a Calcium channel, and two Calcium concentration dependent Potassium ion channels. The dendritic compartment
    integrates injected current inputs as well as any synaptic from other neurons, and elicits a wide calcium spike. This Calcium spike, in turn due to the
    coupling between the two compartments, drives 4-5 tightly chunked sodium ion spikes in the somatic compartment. The model behavior is representative 
    of the bursting behavior observed in HVC(RA) neurons. 
                """)
    model_image = os.path.join(image_dir, "hvcra.png")
    col1, col2 = st.columns([1, 2])
    with col2:
        st.image(model_image, width=400)
    st.markdown("""
                
    With this interactive tool, you get to explore two aspects of model behavior. The first aspect involves how the number of spikes generated in the 
    somatic compartment is constant irrespective of the input strength. The incorporation of temperature dependence exhibits the same result as seen 
    in Hodgkin Huxley model, corresponding to the higher excitability at lower temperatures, and decreased spike frequency. The second aspect involves
    exploring the model behavior under the application of a single excitatory synaptic input. Through this, you can note how neural response changes 
    as a function of input strength, as well as the temperature sensitivity of the response, given by the $Q_{10}$ for rise/response time. The model 
    behavior to an excitatory synaptic input and its temperature dependence has important implications towards emphasizing the role of a neural cicruit 
    localized within HVC towards governing the timing features of birdsong. For more details of the same, please refer to the research work done in the 
    following computational [study](https://www.biorxiv.org/content/10.1101/2025.03.06.641874v1.full.pdf) """) 
    
    st.markdown("""            
    To explore the model behavior, choose either the current input (for number of spikes) or synaptic input (for rise time $Q_{10}$), and then vary the 
    respective input strength from the sidebar. You can also change the $Q_{10}$ values or the temperature and repeat the analysis for model behavior 
    and its temperature dependence as input strength is varied. In both cases, input is made on the dendritic compartment, and the somatic membrane potential 
    is tracked.
    """)

def display_hvci_theory():
    """
    Display the theoretical background for HVC(I) model.
    """

    st.markdown("""
    ## About HVC(I) model

    This neuron model is a conductance-based single compartment model, used to simulate the generation of action potentials
    for an inhibitory class of neurons in the premotor brain region HVC (used as proper name) in songbirds. These neurons exhibit 
    a fast-spiking behavior, and are a major source of inhibition within HVC, which is hypothesised to regulate network activity. 
    This model is a minimal model that breaks the morphology of the HVC(I) neurons into a single somatic compartment. This model 
    is similar to the Hodgkin-Huxley model in terms of its ionic compositions, with the additional incorporation of a high-threshold 
    potassium channel that imparts a fast-spiking behavior to these neurons, representative of the HVC(I) neurons in songbirds.
    """)

    model_image = os.path.join(image_dir, "hvci.png")
    col1, col2 = st.columns([1, 2])
    with col2:
        st.image(model_image, width=400)
    
    st.markdown("""
    With this interactive tool, you get to explore how the spike frequency generated in the somatic compartment changes as a function 
    of the input strength. The incorporation of temperature dependence helps visualize how this feature changes at different temperatures, 
    as well as how a particular input profile that is consistent with experiments shapes the firing rate statistics and its temperature dependence.
    The model behavior explored here has important implications towards the role of HVC(I) neurons and their connectivity patterns with the HVC(RA)
    neurons in imparting temperature robustness to the HVC(RA) burst propagation time. For more details, please refer to the 
    computational [study](https://www.biorxiv.org/content/10.1101/2025.03.06.641874v1.full.pdf)
                
    To explore the model behavior, choose either the current input or synaptic input, and then vary the respective input strength from the sidebar. 
    You can also change the $Q_{10}$ values or the temperature and repeat the analysis for model behavior and its temperature dependence as input 
    strength is varied. 
    """)

def create_current_stimulus_array(time_array, i_amp, i_start, i_end):
    
    """                                             ___
    Function for creating a pulse current input ___|   |___
    
    Parameters
    ----------
    time_array : array_like
        Time vector
    i_amp : float
        The amplitude of the current pulse (muA/cm2).
    i_start : float
        The start time (ms) of the current pulse.
    i_end : float
        The end time (ms) of the current pulse.
    
    Returns
    -------
    numpy.ndarray
        A numpy array containing current input values at different time points.
    """

    current_stimulus = np.zeros_like(time_array)
    start_id = np.argmin(np.abs(time_array - i_start))
    end_id = np.argmin(np.abs(time_array - i_end))
    current_stimulus[start_id:end_id] = i_amp
    return current_stimulus

def create_synapse_stimulus_array(time_array, temp, q_gate, g_max, g_start, step_size):
    """                                            
    Function for creating a synapse stimulus array 
    
    Parameters
    ----------
    time_array : array_like
        Time vector
    temp : float
        Temperature (Celsius)
    q_gate : float
        Q_10 for conformation change dependent processes
    g_max : float
        The size of the synaptic kick (mS/cm2)
    g_start : float
        Time (ms) at which synpatic kick is applied
    step_size : float
        Time step (ms) for the simulation.
    
    Returns
    -------
    numpy.ndarray
        A numpy array containing the profile for the synaptic input.
    """
    synapse_stimulus = np.zeros_like(time_array)
    start_id = np.argmin(np.abs(time_array - g_start))
    q = pow(q_gate, 0.1 * (40.0 - temp))
    for i in range(len(time_array) - start_id):
        synapse_stimulus[start_id + i] = g_max * np.exp(-i * step_size / ( 2 * q ))

    return synapse_stimulus

def generate_poisson_events(event_time, freq):

    """
    Function to update the time of arrival of a Poisson spike

    Parameters
    ----------
    event_time : float
        Last time (ms) at which a spike arrived
    freq : float
        Frequency (Hz) of Poisson process
    
    Returns
    -------
    float
        The latest time of a Poisson spike arrival
    """
    rand_num = np.random.uniform(0.0, 1.0)
    event_time = event_time - (log(rand_num) * 1000 / freq)
   
    return event_time

def create_synapse_stimulus_over_intervals_array(time_array, temp, q_gate, g_max, freq, interval_size, step_size, simulation_time):
    """                                            
    Function for creating a synapse stimulus array where the synaptic input
    is made using a Poisson process over multiple time intervals of fixed size
    
    Parameters
    ----------
    time_array : array_like
        Time vector
    temp : float
        Temperature (Celsius)
    q_gate : float
        Q_10 for conformation change dependent processes
    g_max : float
        The size of the synaptic kick made (mS/cm2)
    freq : float
        Frequency of the Poisson process (Hz)
    interval_size : float
        Size of the interva (ms) over which the input is made
    step_size : float
        Time step (ms) used for the simulation.
    simulation_time : float
        Total duration (ms) of the simulation
    Returns
    -------
    numpy.ndarray
        A numpy array containing the profile for the synaptic input.
    """

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
    Function to assess whether the HVC(RA) model neuron spiked 
    and to return a response time for the soma. The response time 
    is defined as the time taken for somatic membrane potential to reach 
    a peak value from input time for subthreshold response 
    or the time to reach -20 mV for superthreshold response.

    The response time is calculated based off what type of input is applied
    (current pulse or synaptic input) and the onset of that input.

    Parameters
    ----------
    time : array_like
        Time vector
    vd : array_like
        Dendritic membrane potential trace (mV).
    vs : array_like
        Somatic membrane potential trace (mV).
    current_amplitude : float, optional
        Amplitude of the injected current (muA/cm2) (if applicable).
    current_input_start_time : float, optional
        Start time (ms) of the injected current.
    excitatory_input_start_time : float, optional
        Start time (ms) of the excitatory synaptic input.
    excitatory_input_strength : float, optional
        Strength of the excitatory synaptic input (mS/cm2) (if applicable).
    threshold : float, optional
        Voltage threshold (in mV) used to detect a spike. Default is -20 mV.

    Returns
    -------
    float
        The response time of the soma (ms) relative to stimulus onset.
        Returns the time to first spike (threshold crossing) if the neuron spiked,
        or time to peak somatic voltage otherwise.
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

    st.sidebar.header('Model Parameters')        
    st.sidebar.subheader('Current stimulus settings')

    i_amp = st.sidebar.slider('Current amplitude ($\\mu A/cm^2$)', -1.0, 3.0, 0.0, 0.25, key=f'i_amp_{lif_reset_key}')
      
    lif_reset_pressed = st.sidebar.button("Reset")

    if lif_reset_pressed:
        st.session_state.lif_current_list = []
        st.session_state.lif_frequency_list = []
        st.session_state.lif_last_current = 0.0
        st.session_state.lif_reset_counter += 1 

        st.rerun()

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

        with st.spinner(f"Running simulation for {i_amp:.1f} $\\mu A/cm^2$ ..."):

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
            
            st.success(f"Added: {i_amp:.2f} $\\mu A/cm^{2}$ → {frequency:.1f} Hz")
            
    elif current_exists and current_changed:
        st.info(f"Current {i_amp:.1f} $\\mu A/cm^2$ already tested")
    
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
    st.sidebar.subheader('Temperature and $Q_{10}$ values')
    q_gate = st.sidebar.slider('$Q_{10}$ for conformation dependent processes', 2.0, 4.0, 3.0, 0.1, on_change=handle_reset)
    q_cond = st.sidebar.slider('$Q_{10}$ for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1, on_change=handle_reset)

    st.sidebar.subheader('Current stimulus settings')
    i_amp = st.sidebar.slider('Current amplitude ($\\mu A/cm^2$)', 0.0, 15.0, 0.0, 1.0, key=f'i_amp_{hh_reset_key}')
    
    temperature = st.sidebar.selectbox('Temperature in Celsius (control fixed at 6.3 Celsius)', [0.0, 10.0, -10.0], width=300, on_change=handle_reset) # control temperature = 6.3 celsius

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

    v_control= solution_control[:, 0]
    v_alt= solution_alt[:, 0]


    hh_current_list = st.session_state.hh_current_list
    hh_last_current = st.session_state.hh_last_current

    current_changed = abs(i_amp - hh_last_current) >= 0.5
    current_exists = any(abs(c - i_amp) == 0.00 for c in hh_current_list)    

    if current_changed and not current_exists:

        with st.spinner(f"Running F-I simulation for {i_amp:.1f} $\\mu A/cm^{2}$..."):
            spike_indices = []
            spike_indices_ = []
            threshold = -20
            
            solution_control = neuron_control.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
            solution_alt = neuron_alt.simulate(time, STEP_SIZE, current_stimulus_array=current_stimulus)
        
            v_control= solution_control[:, 0]
            v_alt= solution_alt[:, 0]
            st.session_state.hh_current_list.append(i_amp)

            for i in range(1, len(v_control)):
                if v_control[i-1] < threshold and v_control[i] >= threshold:
                    spike_indices.append(i)
                if v_alt[i-1] < threshold and v_alt[i] >= threshold:
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

            
            st.session_state.hh_frequency_list_control.append(frequency_control)
            st.session_state.hh_frequency_list_alt.append(frequency_alt)

            sorted_pairs = sorted(zip(st.session_state.hh_current_list, st.session_state.hh_frequency_list_control, st.session_state.hh_frequency_list_alt))
            st.session_state.hh_current_list, st.session_state.hh_frequency_list_control, st.session_state.hh_frequency_list_alt = zip(*sorted_pairs)
            st.session_state.hh_current_list = list(st.session_state.hh_current_list)
            st.session_state.hh_frequency_list_control = list(st.session_state.hh_frequency_list_control)
            st.session_state.hh_frequency_list_alt = list(st.session_state.hh_frequency_list_alt)
            
            st.success(f"Added: {i_amp:.1f} $\\mu A/cm^{2}$")
    
    elif current_exists and current_changed:
        st.info(f"Current {i_amp:.1f} $\\mu A/cm^2$ already tested")

    st.session_state.hh_last_current = i_amp    

    return v_control, v_alt, time, current_stimulus, temperature, st.session_state.hh_current_list, st.session_state.hh_frequency_list_control, st.session_state.hh_frequency_list_alt, st.session_state.hh_last_current

def create_sidebar_controls_hvcra():
    '''
    Sidebar controls for the HVC(RA) projection neuron model

    Returns:
    -------
    params : dict
    Dictionary containing all parameter values for the HVC(RA)
    '''

    st.sidebar.header('Model Parameters')

    input_type = st.sidebar.selectbox('Input type', ['Current input', 'Synaptic input'], width=200)

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
        st.sidebar.subheader('Temperature and $Q_{10}$ values')

        temperature = st.sidebar.selectbox('Altered Temperature in $^o$C (control set to 40$^o$ C)', [30.0, 35.0], width=300, on_change=handle_reset) #  control temperature = 40.0 celsius
        
        q_gate = st.sidebar.slider('$Q_{10}$ for conformation dependent processes', 2.0, 4.0, 3.0, 0.1, on_change=handle_reset)
        q_cond = st.sidebar.slider('$Q_{10}$ for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1, on_change=handle_reset)

        st.sidebar.subheader('Current stimulus settings')

        i_amp = st.sidebar.slider(f'Current amplitude ($\\mu A/cm^{2}$)', 0.0, 1.0, 0.00, 0.1, key=f'i_amp_{hvcra_reset_key}')

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
        st.sidebar.subheader('Temperature and $Q_{10}$ values')
        temperature = st.sidebar.selectbox('Altered Temperature in $^o$C (control set to 40$^o$ C)', [30.0, 35.0], width=300, on_change=handle_reset) #  control temperature = 40.0 celsius
        
        q_gate = st.sidebar.slider('$Q_{10}$ for conformation dependent processes', 2.0, 4.0, 3.0, 0.1, on_change=handle_reset)
        q_cond = st.sidebar.slider('$Q_{10}$ for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1, on_change=handle_reset)

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

        ge_max = st.sidebar.slider('Excitatory Synapse Strength $(mS/cm^{2})$', 0.01, 1.0, 0.01, 0.01, key=f'ge_max_{hvcra_reset_key}')

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

    input_type = st.sidebar.selectbox('Input type', ['Single current pulse', 'Synaptic input in multiple intervals'])
    
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
        
        st.sidebar.subheader('Temperature and $Q_{10}$ values')
        
        temperature = st.sidebar.selectbox('Temperature in Celsius', [30.0, 35.0], on_change=handle_reset) #  control temperature = 40.0 celsius
        q_gate = st.sidebar.slider('$Q_{10}$ for conformation dependent processes', 2.0, 4.0, 3.0, 0.1, on_change=handle_reset)
        q_cond = st.sidebar.slider('$Q_{10}$ for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1, on_change=handle_reset)

        st.sidebar.subheader('Current stimulus settings')

        i_amp = st.sidebar.slider(f'Current amplitude ($\\mu A/cm^{2}$)', 0.0, 20.0, 0.0, 2.5, key=f'i_amp_{hvci_reset_key}')
        
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

        st.sidebar.subheader('Temperature and $Q_{10}$ values')

        temperature = st.sidebar.selectbox('Temperature in Celsius', [30.0, 35.0], on_change=handle_reset) #  control temperature = 40.0 celsius
        q_gate = st.sidebar.slider('$Q_{10}$ for conformation dependent processes', 2.0, 4.0, 3.0, 0.1, on_change=handle_reset)
        q_cond = st.sidebar.slider('$Q_{10}$ for diffusion dependent processes', 1.0, 2.0, 1.3, 0.1, on_change=handle_reset)

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
    SIMULATION_TIME = 250.0 # ms
    time = np.arange(0, SIMULATION_TIME, STEP_SIZE)

    vs_control = np.full((25000, 1), 0)
    vs_alt = np.full((25000, 1), 0)
    vd_control = np.full((25000, 1), 0)
    vd_alt = np.full((25000, 1), 0)

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
        
        vs_control = solution_control[:, 1]
        vd_control = solution_control[:, 0]

        vs_alt = solution_alt[:, 1]
        vd_alt = solution_alt[:, 0]
        
        hvcra_current_input_list = st.session_state.hvcra_current_input_list
        hvcra_last_current = st.session_state.hvcra_last_current_input

        current_changed = abs(i_amp - hvcra_last_current) > 0.01
        current_exists = any(c == i_amp for c in hvcra_current_input_list)

        if current_changed and not current_exists:

            with st.spinner(f"Running simulation for {i_amp:.2f} $\\mu A/cm^{2}$..."):
                
                solution_control = neuron_control.simulate(time,
                                                        STEP_SIZE,
                                                        current_stimulus_array=current_stimulus
                                                        )
                
                solution_alt = neuron_alt.simulate(time,
                                                STEP_SIZE,
                                                current_stimulus_array=current_stimulus)

                vs_control = solution_control[:, 1]
                vd_control = solution_control[:, 0]

                vs_alt = solution_alt[:, 1]
                vd_alt = solution_alt[:, 0]

                st.success(f"finished running for current input {i_amp:.2f} $\\mu A/cm^{2}$!")

                spike_count_control = 0
                spike_count_alt = 0
                threshold = -20           
        
                for i in range(1, len(vs_control)):
                    if vs_control[i-1] < threshold and vs_control[i] >= threshold:
                        spike_count_control += 1
                    if vs_alt[i-1] < threshold and vs_alt[i] >= threshold:
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
            st.info(f"Current input {i_amp:.2f} $\\mu A/cm^{2}$already tested")
        
        st.session_state.hvcra_last_current_input = i_amp

        return input_type, q_cond, fluctuations, vs_control, vs_alt, vd_control, vd_alt, time, current_stimulus, current_stimulus, temperature, st.session_state.hvcra_current_input_list, st.session_state.hvcra_frequency_control_list, st.session_state.hvcra_frequency_alt_list, st.session_state.hvcra_last_current_input
 
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

        vs_control = solution_control[:, 1]
        vd_control = solution_control[:, 0]

        vs_alt = solution_alt[:, 1]
        vd_alt = solution_alt[:, 0]

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
                
                vs_control = solution_control[:, 1]
                vd_control = solution_control[:, 0]

                vs_alt = solution_alt[:, 1]
                vd_alt = solution_alt[:, 0]

                response_time_control = response_time(time, 
                                                        vd_control, 
                                                        vs_control, 
                                                        excitatory_input_start_time=ge_start, 
                                                        excitatory_input_strength=ge_max)
                
                response_time_alt = response_time(time, 
                                                        vd_alt, 
                                                        vs_alt, 
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
            
        return input_type, q_cond, fluctuations, vs_control, vs_alt, vd_control, vd_alt, time, excitatory_synapse_stimulus_control, excitatory_synapse_stimulus_alt, temperature, st.session_state.hvcra_synaptic_input_list, st.session_state.response_time_control_list, st.session_state.response_time_alt_list, st.session_state.hvcra_last_synaptic_input

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

    v_control= np.full((10000, 1), 0)
    v_alt= np.full((10000, 1), 0)

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
        
        v_control= solution_control[:, 0]
        v_alt= solution_alt[:, 0]
        
        hvci_current_input_list = st.session_state.hvci_current_input_list
        hvci_last_current = st.session_state.hvci_last_current_input

        current_changed = abs(i_amp - hvci_last_current) > 0.01
        current_exists = any(c == i_amp for c in hvci_current_input_list)

        if current_changed and not current_exists:

            with st.spinner(f"Running simulation for {i_amp:.2f} $\\mu A/cm^{2}$..."):

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

                st.success(f"finished running for {i_amp:.2f} $\\mu A/cm^{2}$!")

                v_control= solution_control[:, 0]
                v_alt= solution_alt[:, 0]

                spike_count_control = 0
                spike_count_alt = 0
                threshold = -20           
        
                for i in range(1, len(v_control)):
                    if v_control[i-1] < threshold and v_control[i] >= threshold:
                        spike_count_control += 1
                    if v_alt[i-1] < threshold and v_alt[i] >= threshold:
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
            st.info(f"Current input {i_amp:.2f} $\\mu A/cm^{2}$ already tested")
        
        st.session_state.hvci_last_current_input = i_amp
        
        return input_type, v_control, v_alt, time, current_stimulus, temperature, st.session_state.hvci_current_input_list, st.session_state.hvci_frequency_control_list, st.session_state.hvci_frequency_alt_list, st.session_state.hvci_last_current_input

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
        
        v_control= solution_control[:, 0]
        v_alt= solution_alt[:, 0]
        
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
                
                v_control= solution_control[:, 0]
                v_alt= solution_alt[:, 0]

                spike_count_control = 0
                spike_count_alt = 0
                threshold = -20           
        
                for i in range(1, len(v_control)):
                    if v_control[i-1] < threshold and v_control[i] >= threshold:
                        spike_count_control += 1
                    if v_alt[i-1] < threshold and v_alt[i] >= threshold:
                        spike_count_alt += 1
                    
                frequency_control = spike_count_control #* 1000 / simulation_time
                frequency_alt = spike_count_alt #* 1000 / simulation_time

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
        
        return input_type, v_control, v_alt, time, excitatory_synapse_stimulus_control, temperature, st.session_state.hvci_synaptic_input_list, st.session_state.hvci_frequency_control_list, st.session_state.hvci_frequency_alt_list, st.session_state.hvci_last_synaptic_input
