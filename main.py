import streamlit as st
import matplotlib.pyplot as plt

from utils import (display_lif_theory,
                   prepare_lif_plots,
                   display_hh_theory,
                   prepare_hh_plots,
                   display_hvcra_theory,
                   prepare_hvcra_plots,
                   display_hvci_theory,
                   prepare_hvci_plots,
                   )

from visualization import (plot_lif_membrane_potential,
                           plot_current_frequency_relationship,
                           plot_membrane_potential,
                           plot_somatic_membrane_potential,
                           plot_dendritic_membrane_potential)

st.set_page_config(
    layout = 'wide',
    page_title = 'Neuron model simulator'
)
st.markdown("""
            <style>
            .big-font {
                font-size:25px !important;
            }
            </style>
            """, 
            unsafe_allow_html=True)

st.title('Models for simulating and analyzing the dynamics of a neuron')

with st.expander("Introduction"):
    st.markdown('<p class="big-font"> A neuron is a special type of cell that can generate electrical signals. It does so by utilizing the influx and outflux of a variety of ions ' \
                'through gates or channels that are specific to a given ion. Neuron cells can connect and communicate with each other to transmit this electrical signal to one another, ' \
                ' and patterns of these electrical activity across  connected circuits between the neurons encodes for pretty much any information as well as gives rise to behavior. ' \
                'Neuroscience is an area of study that focuses on understanding the nervous system, which is made up of the brain and the spinal cord, and a neuron is a fundamental' \
                ' unit of this system.</p>', unsafe_allow_html=True) \

    st.image("../streamlit_images/neuron.jpg", width=1000)
        
    st.markdown('<p class="big-font"> In this interactive tool, one can explore different neuron models, that serve as a representation for the dynamics of a neuron. \
                We will look at the simplest model: the leaky-integrate-and-fire neuron model, that does not incorporate any biophysical details, but still gives a good enough insight into the ' \
                'basic dynamics involved. Followed by this, one can explore the Hodgkin-Huxley model, which incorporates some biophysical details that govern action potential generation. ' \
                'Finally one can explore generalized biophysical models for the two classes of neurons found in a brain region called HVC (known as proper name) in songbird species.' \
                'This neurons exhibit a precise signature of activity associated with auditory features of the song. Both the Hodgkin</p>', unsafe_allow_html=True)

#st.markdown('''<p class="big-font"> Click on the button to choose from which neuron model you'd like to learn about </p>''', unsafe_allow_html=True)

Neuron_class = st.selectbox('Choose a Neuron model',
                             ['',
                              'Integrate and Fire model',
                              'Hodgkin Huxley',
                              'HVC neurons'])

if Neuron_class == 'Integrate and Fire model':

    display_lif_theory()

    v, time, current_stimulus, current_list, frequency_list, last_current = prepare_lif_plots()

    tab1, tab2 = st.tabs(['membrane potential', 'current-frequency relationship'])

    with tab1:
        fig_mp = plot_lif_membrane_potential(time,
                                             v,
                                             current_stimulus)
        
        st.pyplot(fig_mp)
    
    with tab2:

        fig_fi = plot_current_frequency_relationship(current_list,
                                                     frequency_list,
                                                     last_current,
                                                     )
     
        st.pyplot(fig_fi)

if Neuron_class == 'Hodgkin Huxley':

    display_hh_theory()
    v, v_, time, current_stimulus, temperature = prepare_hh_plots()

    tab1, tab2 = st.tabs(['membrane potential', 'Gating variables'])

    with tab1:
        fig_mp = plot_membrane_potential(time, 
                                        v, 
                                        v_, 
                                        current_stimulus,
                                        temperature)

        st.pyplot(fig_mp)

elif Neuron_class == 'HVC neurons':
    
    HVC_neuron_type = st.selectbox('HVC neuron', ['Choose HVC neuron type', 'HVC(RA)', 'HVC(I)'])
    
    if HVC_neuron_type == 'HVC(RA)':

        display_hvcra_theory()
        Vs, Vs_, Vd, Vd_, time, temperature, response_time_displayed, response_time_displayed_, response_time_q10 = prepare_hvcra_plots()

        tab1, tab2 = st.tabs(['Soma membrane potential', 'Dendrite membrane potential'])
        
        with tab1:
                
            fig_mp = plot_somatic_membrane_potential(time, 
                                            Vs, 
                                            Vs_, 
                                            temperature, 
                                            response_time_displayed, 
                                            response_time_displayed_, 
                                            response_time_q10)

            st.pyplot(fig_mp)

            with tab2:
                # Plot dendritic membrane potential

                fig_mp = plot_dendritic_membrane_potential(time, 
                                                Vd, 
                                                Vd_, 
                                                temperature, 
                                                )

                st.pyplot(fig_mp)

    elif HVC_neuron_type == 'HVC(I)':

        display_hvci_theory()
        v, v_, time, current_stimulus = prepare_hvci_plots()

        tab1, tab2 = st.tabs(['membrane potential', 'gating variables'])
        
        with tab1:

            fig_mp = plot_membrane_potential(time,
                                             v,
                                             v_,
                                             current_stimulus)
            
            st.pyplot(fig_mp)