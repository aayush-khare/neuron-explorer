import streamlit as st

from utils import (display_introduction,
                   display_lif_theory,
                   prepare_lif_plots,
                   display_hh_theory,
                   prepare_hh_plots,
                   display_hvcra_theory,
                   prepare_hvcra_plots,
                   display_hvci_theory,
                   prepare_hvci_plots,
                   )

from visualization import (plot_lif_membrane_potential,
                           plot_hh_membrane_potential,
                           plot_somatic_membrane_potential,
                           plot_dendritic_membrane_potential,
                           plot_somatic_membrane_potential_with_spike_counts,
                           plot_somatic_membrane_potential_q10,
                           plot_hvci_membrane_potential)

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

Neuron_class = st.selectbox('Contents',
                             ['Introduction',
                              'Integrate and Fire model',
                              'Hodgkin Huxley',
                              'HVC neurons'])

if Neuron_class == 'Introduction':

    display_introduction()
    
if Neuron_class == 'Integrate and Fire model':

    display_lif_theory()

    v, time, current_stimulus, current_list, frequency_list, last_current = prepare_lif_plots()

    fig_fi = plot_lif_membrane_potential(time,
                                         v,
                                         current_stimulus,
                                         current_list,
                                         frequency_list,
                                         last_current)
    
    st.pyplot(fig_fi)

if Neuron_class == 'Hodgkin Huxley':

    display_hh_theory()
    v_control, v_alt, time, current_stimulus, temperature, current_list, frequency_list_control, frequency_list_alt, last_current = prepare_hh_plots()

    tab1, tab2 = st.tabs(['membrane potential and f-I relationship', 'Gating variables'])

    with tab1:
        fig_mp = plot_hh_membrane_potential(time,
                                            v_control,
                                            v_alt,
                                            current_stimulus,
                                            temperature,
                                            current_list,
                                            frequency_list_control,
                                            frequency_list_alt,
                                            last_current)

        st.pyplot(fig_mp)

if Neuron_class == 'HVC neurons':
    
    HVC_neuron_type = st.selectbox('HVC neuron', ['Choose HVC neuron type', 'HVC(RA)', 'HVC(I)'])
    
    if HVC_neuron_type == 'HVC(RA)':

        display_hvcra_theory()
        input_type, fluctuations, vs_control, vs_alt, vd_control, vd_alt, time, temperature, input_list, control_data_list, alt_data_list, last_input = prepare_hvcra_plots()

        if input_type == "Current input":
            tab1, tab2= st.tabs(['Soma membrane potential and spike frequency', 'Dendrite membrane potential'])

            with tab1:
                
                fig_mp = plot_somatic_membrane_potential_with_spike_counts(time, 
                                                                           vs_control,
                                                                           vs_alt,
                                                                           temperature,
                                                                           input_list,
                                                                           control_data_list,
                                                                           alt_data_list,
                                                                           last_input)

                st.pyplot(fig_mp)

            with tab2:

                fig_mp = plot_dendritic_membrane_potential(time,
                                                           vd_control,
                                                           vd_alt,
                                                           temperature
                                                           )

                st.pyplot(fig_mp)

        elif input_type == "Synaptic input":
            if fluctuations == 'off':
                tab1, tab2 = st.tabs(['Soma membrane potential and response time Q10', 'Dendrite membrane potential'])

                with tab1:

                    fig_a = plot_somatic_membrane_potential_q10(time, 
                                                                vs_control, 
                                                                vs_alt, 
                                                                temperature, 
                                                                input_list, 
                                                                control_data_list, 
                                                                alt_data_list,
                                                                last_input
                                                                )

                    st.pyplot(fig_a)

                with tab2:

                    fig_mp = plot_dendritic_membrane_potential(time,
                                                               vd_control,
                                                               vd_alt,
                                                               temperature
                                                               )

                    st.pyplot(fig_mp)
            
            else:
                tab1, tab2 = st.tabs(['Soma membrane potential', 'Dendrite membrane potential'])

                with tab1:

                    fig_mp = plot_somatic_membrane_potential(time, 
                                                             vs_control, 
                                                             vs_alt, 
                                                             temperature
                                                            )

                    st.pyplot(fig_mp)

                with tab2:

                    fig_mp = plot_dendritic_membrane_potential(time,
                                                               vd_control,
                                                               vd_alt,
                                                               temperature
                                                               )

                    st.pyplot(fig_mp)

    elif HVC_neuron_type == 'HVC(I)':

        display_hvci_theory()
        v_control, v_alt, time, current_stimulus, temperature, current_list, frequency_list_control, frequency_list_alt, last_current = prepare_hvci_plots()

        tab1, tab2 = st.tabs(['membrane potential', 'gating variables'])
        
        with tab1:

            fig_mp = plot_hvci_membrane_potential(time,
                                                  v_control,
                                                  v_alt,
                                                  current_stimulus,
                                                  temperature,
                                                  current_list, 
                                                  frequency_list_control,
                                                  frequency_list_alt,
                                                  last_current
                                                  )
            
            st.pyplot(fig_mp)