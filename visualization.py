import numpy as np
import matplotlib.pyplot as plt

def plot_somatic_membrane_potential_with_spike_counts(time, vs_control, vs_alt, temperature, input_list, control_data_list, alt_data_list, last_input):
    """
    Plots the somatic membrane potential with spike counts for two conditions.
    Args:
        time: Time vector.
        vs_control: Membrane potential for the control condition.
        vs_alt: Membrane potential for the alternative condition.
        temperature: Temperature for the alternative condition.
        input_list: List of current inputs.
        control_data_list: Spike counts for the control condition.
        alt_data_list: Spike counts for the alternative condition.
        last_input: The last current input value entered.

    Returns:
        A matplotlib figure containing the plots.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(time, vs_control, 'r', label='40.0$^o$ C')
    ax1.plot(time, vs_alt, 'b', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_xlim(100, 200)
    ax1.set_ylim(-100, 100)
    ax1.grid(True, alpha=0.3)

    ax1.set_title('Membrane Potential for the somatic compartment')

    if len(input_list) > 0:
        sorted_pairs = sorted(zip(input_list, control_data_list, alt_data_list))
        sorted_input, sorted_control_data, sorted_alt_data = zip(*sorted_pairs)
        sorted_input = list(sorted_input)
        sorted_control_data = list(sorted_control_data)
        sorted_alt_data = list(sorted_alt_data)

        ax2.plot(sorted_input, sorted_control_data,
                'ro-', linewidth=1, markersize=4, alpha=0.7)
        ax2.plot(sorted_input, sorted_alt_data,
                'bo-', linewidth=1, markersize=4, alpha=0.7)
        
        if last_input in [c for c in sorted_input if c == last_input]:
            idx = next(i for i, c in enumerate(sorted_input) if c == last_input)
            current_control_data = sorted_control_data[idx]
            current_alt_data = sorted_alt_data[idx]
            ax2.plot(last_input, current_control_data, 'o', color='red', markersize=10,
                    label=f'{last_input:.1f} $\mu A/cm^{2}$, {current_control_data} spikes')
            ax2.plot(last_input, current_alt_data, 'o', color='blue', markersize=10,
                    label=f'{last_input:.1f} $\mu A/cm^{2}$, {current_alt_data} spikes')
        
        ax2.set_xlabel(f'Current input ($\mu A/cm^{2}$)', fontsize=12)
        ax2.set_ylabel('num spikes', fontsize=12)
        ax2.set_title('Somatic spikes vs current injected', fontsize=14)
        ax2.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_ylim(-0.5, 5.5)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
    else:
        ax2.set_xlabel(f'Current input ($\mu A/cm^{2}$)', fontsize=12)
        ax2.set_ylabel('num spikes', fontsize=12)
        ax2.set_title('Somatic spikes vs current injected', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_somatic_membrane_potential_q10(time, vs_control, vs_alt, temperature, synaptic_input_list, response_time_control_list, responase_time_alt_list, last_synaptic_input):
    """
    Plots the somatic membrane potential $Q_{10}$ for two conditions.

    Args:
        time: Time vector.
        vs_control: Membrane potential for the control condition.
        vs_alt: Membrane potential for the alternative condition.
        temperature: Temperature for the alternative condition.
        synaptic_input_list: List of synaptic inputs.
        response_time_control_list: Response times for the control condition.
        response_time_alt_list: Response times for the alternative condition.
        last_synaptic_input: The last synaptic input value entered.

    Returns:
        A matplotlib figure containing the plots.

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(time, vs_control, 'red', label='40.0$^o$ C')
    ax1.plot(time, vs_alt, 'blue', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)', color='black')
    ax1.set_xlim(100, 200)
    ax1.set_ylim(-100, 100)

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    ax1.set_title('Membrane Potential for the somatic compartment')

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')

    if len(synaptic_input_list) > 0:
        sorted_pairs = sorted(zip(synaptic_input_list, response_time_control_list, responase_time_alt_list))
        sorted_synaptic_input, sorted_response_control, sorted_response_alt = zip(*sorted_pairs)
        sorted_synaptic_input = list(sorted_synaptic_input)
        sorted_response_control = list(sorted_response_control)
        sorted_response_alt = list(sorted_response_alt)

        sorted_response_q = [(r_ / r) ** (0.1 * (40.0 - temperature)) for r, r_ in zip(sorted_response_control, sorted_response_alt)]
        ax2.plot(sorted_synaptic_input, sorted_response_q,
                'go-', linewidth=1, markersize=4, alpha=0.7, label='Measured points')
        
        if last_synaptic_input in [c for c in sorted_synaptic_input if c == last_synaptic_input]:
            idx = next(i for i, c in enumerate(sorted_synaptic_input) if c == last_synaptic_input)
            current_q10 = sorted_response_q[idx]
            ax2.plot(last_synaptic_input, current_q10, 'o', color='orange', markersize=8,
                    label = fr'gE: {last_synaptic_input} mS/cm$^2$, ' + r'$Q_{10}$' + f' = {current_q10:.2f}')
        
        ax2.set_xlabel('Excitatory synaptic input, gE (mS/cm$^2$)', fontsize=12)
        ax2.set_ylabel('$Q_{10}$', fontsize=12)
        ax2.set_title('Somatic response time $Q_{10}$', fontsize=14)
        ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        if len(synaptic_input_list) > 1:
            ax2.set_xlim(0.0, 1.01)
            ax2.set_ylim(0, 4)
    else:
        ax2.set_xlabel('Excitatory synaptic input, gE (mS/cm$^2$)', fontsize=12)
        ax2.set_ylabel('$Q_{10}$', fontsize=12)
        ax2.set_title('Somatic response time $Q_{10}$', fontsize=14)
        ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        
    plt.tight_layout()
    return fig

def plot_somatic_membrane_potential(time, vs_control, vs_alt, temperature):
    """
    Plots the somatic membrane potential for two different temperatures.

    Args:
        time: Time vector.
        vs_control: Membrane potential for the control condition.
        vs_alt: Membrane potential for the alternative condition.
        temperature: Temperature for the alternative condition.

    Returns:
        A matplotlib figure containing the plots.
    """

    fig, ax1 = plt.subplots(figsize=(15, 6))

    ax1.plot(time, vs_control, 'red', label='40.0$^o$ C')
    ax1.plot(time, vs_alt, 'blue', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)', color='black')
    ax1.set_ylim(-150, 150)
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    ax1.set_title('Membrane Potential for the somatic compartment')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_dendritic_membrane_potential(time, vd_control, vd_alt, temperature):

    """
    Plots the dendritic membrane potential for two different temperatures.

    Args:
        time: Time vector.
        vd_control: Membrane potential for the control condition.
        vd_alt: Membrane potential for the alternative condition.
        temperature: Temperature for the alternative condition.

    Returns:
        A matplotlib figure containing the plots.
    """
    fig, ax1 = plt.subplots(figsize=(15, 6))

    ax1.plot(time, vd_control, 'red', label='40.0$^o$ C')
    ax1.plot(time, vd_alt, 'blue', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)', color='black')
    ax1.set_ylim(-150, 150)
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    ax1.set_title('Membrane Potential for the dendritic compartment')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_hvci_membrane_potential(time, v_control, v_alt, stimulus, temperature, current_list, frequency_list_control, frequency_list_alt, last_current):

    """
    Plots the membrane potential for the HVCI neuron.

    Args:
        time: Time vector.
        v_control: Membrane potential for the control condition.
        v_alt: Membrane potential for the alternative condition.
        stimulus: Current stimulus.
        temperature: Temperature for the alternative condition.
        current_list: List of injected currents.
        frequency_list_control: Firing frequencies for the control condition.
        frequency_list_alt: Firing frequencies for the alternative condition.
        last_current: The last injected current.

    Returns:
        A matplotlib figure containing the plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax3 = ax1.twinx()
    ax3.plot(time, stimulus, 'tab:green')
    ax3.set_ylabel(f'current stimulus ($\mu A/cm^{2})$', color='tab:green', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.set_ylim(-3.5, 15.5)    


    ax1.plot(time, v_control, 'r', label='40.0$^o$ C')
    ax1.plot(time, v_alt, 'b', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax1.set_ylim(-90, 50)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Membrane Potential and Stimulus Current')

    if len(current_list) > 0:
        sorted_pairs = sorted(zip(current_list, frequency_list_control, frequency_list_alt))
        sorted_current, sorted_frequency_control, sorted_frequency_alt = zip(*sorted_pairs)
        sorted_current = list(sorted_current)
        sorted_frequency_control = list(sorted_frequency_control)
        sorted_frequency_alt = list(sorted_frequency_alt)
        
        ax2.plot(sorted_current, sorted_frequency_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7)
        ax2.plot(sorted_current, sorted_frequency_alt,
                 'bo-', linewidth=1, markersize=4, alpha=0.7)
        
        if last_current in [c for c in sorted_current if c == last_current]:
            idx = next(i for i, c in enumerate(sorted_current) if c == last_current)
            current_freq_control = sorted_frequency_control[idx]
            current_freq_alt = sorted_frequency_alt[idx]
            ax2.plot(last_current, current_freq_control, 'o', color='red', markersize=10,
                    label=f'{last_current:.1f} $\mu A/cm^{2}$, {current_freq_control:.1f} Hz')
            ax2.plot(last_current, current_freq_alt, 'o', color='blue', markersize=10,
                    label=f'{last_current:.1f} $\mu A/cm^{2}$, {current_freq_alt:.1f} Hz')   
        
        ax2.set_xlabel('Injected Current ($\mu A/cm^{2}$)', fontsize=12)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=12)
        ax2.set_title('Current-Frequency (F-I) Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
      
    else:
        ax2.set_xlabel('Injected Current ($\mu A/cm^{2}$)', fontsize=12)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=12)
        ax2.set_title('Current-Frequency (F-I) Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)


    plt.tight_layout()
    return fig

def plot_hh_membrane_potential(time, v_control, v_alt, stimulus, temperature, current_list, frequency_list_control, frequency_list_alt, last_current):
    """
    Plots the membrane potential for the Hodgkin-Huxley neuron model.

    Args:
        time: Time vector.
        v_control: Membrane potential for the control condition.
        v_alt: Membrane potential for the alternative condition.
        stimulus: Current stimulus.
        temperature: Temperature for the alternative condition.
        current_list: List of injected currents.
        frequency_list_control: Firing frequencies for the control condition.
        frequency_list_alt: Firing frequencies for the alternative condition.
        last_current: The last injected current.

    Returns:
        A matplotlib figure containing the plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax3 = ax1.twinx()
    ax3.plot(time, stimulus, 'tab:green')
    ax3.set_ylabel(f'current stimulus ($\mu A/cm^{2})$', color='tab:green', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.set_ylim(-3.5, 15.5)    


    ax1.plot(time, v_control, 'r', label='6.3$^o$ C')
    ax1.plot(time, v_alt, 'b', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax1.set_ylim(-90, 50)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Membrane Potential and Stimulus Current')

    if len(current_list) > 0:
        sorted_pairs = sorted(zip(current_list, frequency_list_control, frequency_list_alt))
        sorted_current, sorted_frequency_control, sorted_frequency_alt = zip(*sorted_pairs)
        sorted_current = list(sorted_current)
        sorted_frequency_control = list(sorted_frequency_control)
        sorted_frequency_alt = list(sorted_frequency_alt)
        
        ax2.plot(sorted_current, sorted_frequency_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7)
        ax2.plot(sorted_current, sorted_frequency_alt,
                 'bo-', linewidth=1, markersize=4, alpha=0.7)
        
        if last_current in [c for c in sorted_current if c == last_current]:
            idx = next(i for i, c in enumerate(sorted_current) if c == last_current)
            current_freq_control = sorted_frequency_control[idx]
            current_freq_alt = sorted_frequency_alt[idx]
            ax2.plot(last_current, current_freq_control, 'o', color='red', markersize=10,
                    label=f'{last_current:.1f} $\mu A/cm^{2}$, {current_freq_control:.1f} Hz')
            ax2.plot(last_current, current_freq_alt, 'o', color='blue', markersize=10,
                    label=f'{last_current:.1f} $\mu A/cm^{2}$, {current_freq_alt:.1f} Hz')   
        
        ax2.set_xlabel('Injected Current ($\mu A/cm^{2}$)', fontsize=12)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=12)
        ax2.set_title('Current-Frequency (F-I) Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
      
    else:
        ax2.set_xlabel('Injected Current ($\mu A/cm^{2}$)', fontsize=12)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=12)
        ax2.set_title('Current-Frequency (F-I) Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)


    plt.tight_layout()
    return fig

def plot_lif_membrane_potential(time, v, current, current_list, frequency_list, last_current):
    """
    Plots the membrane potential for the LIF neuron model.

    Args:
        time: Time vector.
        v: Membrane potential.
        current: Current stimulus.
        current_list: List of injected currents.
        frequency_list: Firing frequencies.
        last_current: The last injected current.

    Returns:
        A matplotlib figure containing the plots.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(time, v, 'b')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)', color='b')
    ax1.set_ylim(-90, 20)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Membrane Potential and Stimulus Current')

    ax3 = ax1.twinx()

    ax3.plot(time, current, 'tab:pink')
    ax3.set_ylabel(f'current stimulus ($\mu A/cm^{2})$', color='tab:pink')
    ax3.tick_params(axis='y', labelcolor='tab:pink')
    ax3.set_ylim(-1.5, 3.5)

    if len(current_list) > 0:
        sorted_pairs = sorted(zip(current_list, frequency_list))
        sorted_current, sorted_frequency = zip(*sorted_pairs)
        sorted_current = list(sorted_current)
        sorted_frequency = list(sorted_frequency)
        
        ax2.plot(sorted_current, sorted_frequency, 
                'o-', color='black', linewidth=1, markersize=4, alpha=1, label='Measured points')
        
        if last_current in [c for c in sorted_current if c == last_current]:
            idx = next(i for i, c in enumerate(sorted_current) if c == last_current)
            current_freq = sorted_frequency[idx]
            ax2.plot(last_current, current_freq, 'ro', markersize=6,
                    label=f'{last_current:.2f} $\mu A/cm^{2}$, {current_freq:.1f} Hz')
        
        ax2.set_xlabel(f'Injected Current ($\mu A/cm^{2}$)', fontsize=10)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=10)
        ax2.set_title('Current-Frequency (F-I) Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        if len(current_list) > 1:
            ax2.set_xlim(min(current_list) - 0.5, max(current_list) + 0.5)
            ax2.set_ylim(-10, max(frequency_list) + 10)

    else:

        ax2.set_xlabel('Injected Current (nA)', fontsize=10)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=10)
        ax2.set_title('Current-Frequency (F-I) Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_phase_plots(V, m, h, n, dt):
    """
    Create phase plots for visualizing neuron dynamics
    
    Parameters:
    -----------
    V, m, h, n : array-like
        Membrane potential and gate variables
    dt : float
        Time step (ms)
        
    Returns:
    --------
    figs : list
        List of figure objects
    """
    figs = []
    
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot(V, n, 'g')
    ax1.set_xlabel('Membrane Potential (mV)')
    ax1.set_ylabel('n (K⁺ activation)')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('K⁺ Activation Phase Plot')
    plt.tight_layout()
    figs.append(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(V, m, 'r')
    ax2.set_xlabel('Membrane Potential (mV)')
    ax2.set_ylabel('m (Na⁺ activation)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Na⁺ Activation Phase Plot')
    plt.tight_layout()
    figs.append(fig2)
    
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.plot(V, h, 'b')
    ax3.set_xlabel('Membrane Potential (mV)')
    ax3.set_ylabel('h (Na⁺ inactivation)')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Na⁺ Inactivation Phase Plot')
    plt.tight_layout()
    figs.append(fig3)
    
    dVdt = np.gradient(V, dt)
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    ax4.plot(V[:-1], dVdt[:-1], 'k')
    ax4.set_xlabel('Membrane Potential (mV)')
    ax4.set_ylabel('dV/dt (mV/ms)')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Phase Plane Plot')
    plt.tight_layout()
    figs.append(fig4)
    
    return figs

def response_time(time, vd, vs, current_amplitude=None, current_input_start_time=None, excitatory_input_start_time=None, excitatory_input_strength=None, threshold=-20):
    """
    Assess whether the neuron spiked or not and return a response time
    The response time is defined as the time to peak somatic membrane 
    potential in case the peak is below threshold or the time to threshold
    from input time

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