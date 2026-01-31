import streamlit as st

from utils import (display_introduction,
                   display_electrical_properties,
                   display_lif_theory,
                   display_lif_summary,
                   prepare_lif_plots,
                   display_hh_theory,
                   prepare_hh_plots,
                   display_hh_summary,
                   display_hvc_background,
                   display_hvcra_theory,
                   prepare_hvcra_plots,
                   display_hvci_theory,
                   prepare_hvci_plots,
                   )

from visualization import (plot_lif_membrane_potential,
                           plot_hh_membrane_potential,
                           plot_hh_membrane_potential_spike_characteristics,
                           plot_hvcra_somatic_spike_characteristics,
                           plot_hvci_spike_characteristics,
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

select_page = st.sidebar.selectbox('Contents',
                             ['Introduction',
                              'Electrical Properties of a Neuron',
                              'Integrate and Fire model',
                              'Hodgkin Huxley',
                              'HVC neurons',
                              'Coming up!'])

if select_page == 'Introduction':
    
    st.title('Neuron model simulator')
    display_introduction()

if select_page == 'Electrical Properties of a Neuron':
    
    st.title('Neuron Biophysics')
    display_electrical_properties()
    
if select_page == 'Integrate and Fire model':

    st.title('Leaky Integrate and Fire (LIF) model')
    select = st.sidebar.selectbox('LIF model contents', ['Theory', 'Plots'])

    if select == 'Theory':

        display_lif_theory()

    else:
        (v, 
         time, 
         current_stimulus, 
         current_list, 
         frequency_list, 
         last_current
          ) = prepare_lif_plots()

        fig_fi = plot_lif_membrane_potential(time,
                                            v,
                                            current_stimulus,
                                            current_list,
                                            frequency_list,
                                            last_current)
        
        st.pyplot(fig_fi)

        display_lif_summary()

if select_page == 'Hodgkin Huxley':

    st.title('Hodgkin-Huxley (HH) model')
    select = st.sidebar.selectbox('HH model contents', ['Theory', 'Plots'])

    if select == 'Theory':
        display_hh_theory()
    
    else:
        (v_control, 
         v_alt, 
         time, 
         current_stimulus, 
         temperature, 
         current_list, 
         frequency_list_control, 
         frequency_list_alt, 
         spike_width_list_control, 
         spike_width_list_alt, 
         isi_list_control, 
         isi_list_alt, 
         last_current 
         ) = prepare_hh_plots()

        st.markdown("## Hodgkin Huxley model: Membrane potential and F-I relationship")

        tab1, tab2 = st.tabs(['Membrane Potential', 'Spike characteristics'])

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
        
        with tab2:
            
            fig_mp = plot_hh_membrane_potential_spike_characteristics(temperature,
                                                                 current_list,
                                                                 spike_width_list_control,
                                                                 spike_width_list_alt,
                                                                 isi_list_control,
                                                                 isi_list_alt,
                                                                 last_current)
            
            st.pyplot(fig_mp)  

        display_hh_summary()  

if select_page == 'HVC neurons':

    st.title('HVC neuron model')
    select = st.sidebar.selectbox('HVC model contents', ['HVC background', 'HVC models'])

    if select == 'HVC background':
        display_hvc_background()
    
    else:
        HVC_neuron_type = st.sidebar.selectbox('HVC neuron', ['Choose HVC neuron type', 'HVC(RA)', 'HVC(I)'])
        
        if HVC_neuron_type == 'HVC(RA)':

            select_ = st.sidebar.selectbox('HVC(RA) model contents', ['Theory', 'Plots'])

            if select_ == 'Theory':
                display_hvcra_theory()
            
            else:
                (input_type, 
                 q_cond, 
                 fluctuations, 
                 vs_control, 
                 vs_alt, 
                 vd_control, 
                 vd_alt, 
                 time, 
                 input_array, 
                 input_array_alt, 
                 temperature, 
                 input_list, 
                 control_data_list, 
                 alt_data_list, 
                 control_spike_width, 
                 alt_spike_width, 
                 control_isi, 
                 alt_isi, 
                 last_input
                  ) = prepare_hvcra_plots()

                if input_type == "Current input":
                    tab1, tab2, tab3 = st.tabs(['Soma membrane potential and spike counts', 'Somatic spike characteristics', 'Dendrite membrane potential'])

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

                        fig_mp = plot_hvcra_somatic_spike_characteristics(temperature,
                                                                          input_list,
                                                                          input_type,
                                                                          control_spike_width,
                                                                          alt_spike_width,
                                                                          control_isi,
                                                                          alt_isi,
                                                                          last_input)

                        st.pyplot(fig_mp)

                    with tab3:

                        fig_mp = plot_dendritic_membrane_potential(time,
                                                                vd_control,
                                                                vd_alt,
                                                                temperature
                                                                )

                        st.pyplot(fig_mp)

                elif input_type == "Synaptic input":
                    if fluctuations == 'off':
                        tab1, tab2, tab3 = st.tabs(['Soma membrane potential and response time Q10', 'Somatic spike characteristics', 'Dendrite membrane potential'])

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

                            fig_mp = plot_hvcra_somatic_spike_characteristics(temperature,
                                                                            input_list,
                                                                            input_type,
                                                                            control_spike_width,
                                                                            alt_spike_width,
                                                                            control_isi,
                                                                            alt_isi,
                                                                            last_input)

                            st.pyplot(fig_mp)

                        with tab3:

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

            select_ = st.sidebar.selectbox('HVC(I) model contents', ['Theory', 'Plots'])

            if select_ == 'Theory':
                display_hvci_theory()

            else:
                (input_type, 
                 q_cond, 
                 v_control, 
                 v_alt, 
                 time, 
                 input_profile_control, 
                 input_profile_alt, 
                 temperature, 
                 input_strength_list, 
                 frequency_list_control, 
                 frequency_list_alt, 
                 control_spike_width, 
                 alt_spike_width, 
                 control_isi, 
                 alt_isi, 
                 last_input_strength
                  ) = prepare_hvci_plots()
                
                if input_type == 'Single current pulse':
                
                    tab1, tab2 = st.tabs(['Membrane potential', 'Spike characteristics'])

                    with tab1:
                        fig_mp = plot_hvci_membrane_potential(time,
                                                            v_control,
                                                            v_alt,
                                                            input_profile_control,
                                                            input_profile_alt,
                                                            q_cond,
                                                            temperature,
                                                            input_strength_list, 
                                                            frequency_list_control,
                                                            frequency_list_alt,
                                                            last_input_strength,
                                                            input_type
                                                            )
                        
                        st.pyplot(fig_mp)
                    
                    with tab2:
                        fig_mp = plot_hvci_spike_characteristics(temperature,
                                                                input_strength_list,
                                                                control_spike_width,
                                                                alt_spike_width,
                                                                control_isi,
                                                                alt_isi,
                                                                last_input_strength)

                        st.pyplot(fig_mp)
                
                else:
                    fig_mp = plot_hvci_membrane_potential(time,
                                                          v_control,
                                                          v_alt,
                                                          input_profile_control,
                                                          input_profile_alt,
                                                          q_cond,
                                                          temperature,
                                                          input_strength_list, 
                                                          frequency_list_control,
                                                          frequency_list_alt,
                                                          last_input_strength,
                                                          input_type
                                                          )
                    
                    st.pyplot(fig_mp)

if select_page == 'Coming up!':
    st.markdown(""" Thank you for going over this interactive neuron model simulator! If you have any feedback to share, do reach out to me at aayushkhare@psu.edu 
                or aayushkhare95@gmail.com. And there are more interactive tools catering to computational neuroscience coming up, so keep checking this 
                streamlit page!""")