import streamlit as st

from utils import (display_introduction,
                   display_electrical_properties,
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

st.title('Neuron model simulator')

select_page = st.selectbox('Contents',
                             ['Introduction',
                              'Electrical Properties of a Neuron',
                              'Integrate and Fire model',
                              'Hodgkin Huxley',
                              'HVC neurons',
                              'Coming up!'])

if select_page == 'Introduction':

    display_introduction()

if select_page == 'Electrical Properties of a Neuron':

    display_electrical_properties()
    
if select_page == 'Integrate and Fire model':

    display_lif_theory()

    v, time, current_stimulus, current_list, frequency_list, last_current = prepare_lif_plots()

    fig_fi = plot_lif_membrane_potential(time,
                                         v,
                                         current_stimulus,
                                         current_list,
                                         frequency_list,
                                         last_current)
    
    st.pyplot(fig_fi)

if select_page == 'Hodgkin Huxley':

    display_hh_theory()
    v_control, v_alt, time, current_stimulus, temperature, current_list, frequency_list_control, frequency_list_alt, last_current = prepare_hh_plots()

    st.markdown("## Hodgkin Huxley model: Membrane potential and F-I relationship")

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

if select_page == 'HVC neurons':
    
    HVC_neuron_type = st.selectbox('HVC neuron', ['Choose HVC neuron type', 'HVC(RA)', 'HVC(I)'])
    
    if HVC_neuron_type == 'HVC(RA)':

        display_hvcra_theory()
        input_type, q_cond, fluctuations, vs_control, vs_alt, vd_control, vd_alt, time, input_array, input_array_alt, temperature, input_list, control_data_list, alt_data_list, last_input = prepare_hvcra_plots()

        if input_type == "Current input":
            tab1, tab2= st.tabs(['Soma membrane potential and spike frequency', 'Dendrite membrane potential'])

            with tab1:
                
                fig_mp = plot_somatic_membrane_potential_with_spike_counts(time, 
                                                                           vs_control,
                                                                           vs_alt,
                                                                           temperature,
                                                                           input_array,
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
                                                                input_array, 
                                                                input_array_alt,
                                                                q_cond,
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
        input_type, v_control, v_alt, time, input_profile, temperature, input_strength_list, frequency_list_control, frequency_list_alt, last_input_strength = prepare_hvci_plots()
        

        fig_mp = plot_hvci_membrane_potential(time,
                                            v_control,
                                            v_alt,
                                            input_profile,
                                            temperature,
                                            input_strength_list, 
                                            frequency_list_control,
                                            frequency_list_alt,
                                            last_input_strength,
                                            input_type
                                            )
        
        st.pyplot(fig_mp)

if select_page == 'Coming up!':
    st.markdown(""" Thank you for going over this interactive neuron model simulator! If you have any feedback to share, do reach out to me. 
                And I will be continuing to work on more such interactive tools catering to computational neuroscience, so keep checking this streamlit page""")