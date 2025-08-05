import numpy as np
import matplotlib.pyplot as plt

def plot_somatic_membrane_potential_with_spike_frequency(time, Vs, Vs_, temperature, current_input_list, soma_spike_frequency, last_current_input):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(time, Vs, 'r', label='40.0$^o$ C')
    ax1.plot(time, Vs_, 'b', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_xlim(100, 200)
    ax1.set_ylim(-100, 100)

    ax1.set_title('Membrane Potential for the somatic compartment')

    if len(current_input_list) > 0:
        sorted_pairs = sorted(zip(current_input_list, soma_spike_frequency))
        sorted_synaptic_input, sorted_response_q = zip(*sorted_pairs)
        sorted_synaptic_input = list(sorted_synaptic_input)
        sorted_response_q = list(sorted_response_q)

        ax2.plot(sorted_synaptic_input, sorted_response_q,
                'go-', linewidth=1, markersize=4, alpha=0.7, label='Measured points')
        
        if last_current_input in [c for c in sorted_synaptic_input if c == last_current_input]:
            idx = next(i for i, c in enumerate(sorted_synaptic_input) if c == last_current_input)
            current_q10 = sorted_response_q[idx]
            ax2.plot(last_current_input, current_q10, 'o', color='orange', markersize=4,
                    label = fr'gE: {last_current_input} mS/cm$^2$, ' + r'$Q_{10}$' + f' = {current_q10:.2f}')
        
        ax2.set_xlabel('Excitatory synaptic input, gE (mS/cm$^2$)', fontsize=12)
        ax2.set_ylabel('$Q_{10}$', fontsize=12)
        ax2.set_title('Somatic response time $Q_{10}$', fontsize=14)
        ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        #if len(current_input_list) > 1:
            #ax2.set_xlim(0.0, 1.01)
            #ax2.set_ylim(0, 4)
    return fig

def plot_somatic_membrane_potential_q10(time, Vs, Vs_, temperature, synaptic_input_list, response_time_q_list, last_synaptic_input):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(time, Vs, 'red', label='40.0$^o$ C')
    ax1.plot(time, Vs_, 'blue', label=f'{temperature}$^o$ C')
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
        sorted_pairs = sorted(zip(synaptic_input_list, response_time_q_list))
        sorted_synaptic_input, sorted_response_q = zip(*sorted_pairs)
        sorted_synaptic_input = list(sorted_synaptic_input)
        sorted_response_q = list(sorted_response_q)

        ax2.plot(sorted_synaptic_input, sorted_response_q,
                'go-', linewidth=1, markersize=4, alpha=0.7, label='Measured points')
        
        if last_synaptic_input in [c for c in sorted_synaptic_input if c == last_synaptic_input]:
            idx = next(i for i, c in enumerate(sorted_synaptic_input) if c == last_synaptic_input)
            current_q10 = sorted_response_q[idx]
            ax2.plot(last_synaptic_input, current_q10, 'o', color='orange', markersize=4,
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

    plt.tight_layout()
    return fig

def plot_somatic_membrane_potential(time, Vs, Vs_, temperature, response_time_displayed=None, response_time_displayed_=None, response_time_q10=None):

    fig, ax1 = plt.subplots(figsize=(15, 6))
    # Plot membrane potential
    ax1.plot(time, Vs, 'red', label='40.0$^o$ C')
    ax1.plot(time, Vs_, 'blue', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)', color='black')
    ax1.set_ylim(-150, 150)
    if response_time_displayed is not None:
        ax1.text(20, -30, f'rise time control: {response_time_displayed} ms', color='red')
        ax1.text(20, -40, f'rise time alt: {response_time_displayed_} ms', color='blue')
        ax1.text(20, -50, f'rise time Q\u2081\u2080: {response_time_q10:.3f}', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Set title and legends
    ax1.set_title('Membrane Potential for the somatic compartment')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_dendritic_membrane_potential(time, Vd, Vd_, temperature):

    fig, ax1 = plt.subplots(figsize=(15, 6))
    # Plot membrane potential
    ax1.plot(time, Vd, 'red', label='40.0$^o$ C')
    ax1.plot(time, Vd_, 'blue', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)', color='black')
    ax1.set_ylim(-150, 150)
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Set title and legends
    ax1.set_title('Membrane Potential for the dendritic compartment')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_soma_response_q10(synaptic_input_list, response_time_q_list, last_synaptic_input):
    
    fig_fi, ax = plt.subplots(figsize=(10, 6))
        
    if len(synaptic_input_list) > 0:
        
        sorted_pairs = sorted(zip(synaptic_input_list, response_time_q_list))
        sorted_synaptic_input, sorted_response_q = zip(*sorted_pairs)
        sorted_synaptic_input = list(sorted_synaptic_input)
        sorted_response_q = list(sorted_response_q)

        ax.plot(sorted_synaptic_input, sorted_response_q,
                'bo-', linewidth=2, markersize=8, alpha=0.7, label='Measured points')
        
        # Highlight current point using sorted data
        if last_synaptic_input in [c for c in sorted_synaptic_input if c == last_synaptic_input]:
            idx = next(i for i, c in enumerate(sorted_synaptic_input) if c == last_synaptic_input)
            current_q = sorted_response_q[idx]
            ax.plot(last_synaptic_input, current_q, 'ro', markersize=12,
                   label=f'Current: {last_synaptic_input:.1f} nA, {current_q:.1f} Hz')
    
    ax.set_xlabel('Injected Current (nA)', fontsize=12)
    ax.set_ylabel('Firing Frequency (Hz)', fontsize=12)
    ax.set_title('Current-Frequency (F-I) Relationship', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if len(synaptic_input_list) > 1:
        ax.set_xlim(-0.1, max(synaptic_input_list) + 0.2)
        ax.set_ylim(0, max(response_time_q_list) + 2)
    
    return fig_fi

def plot_membrane_potential(time, V, V_, stimulus, temperature, current_list, frequency_list_control, frequency_list_alt, last_current):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax3 = ax1.twinx()
    ax3.plot(time, stimulus, 'tab:pink')
    ax3.set_ylabel(f'current stimulus ($\mu A/cm^{2})$', color='tab:pink', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='tab:pink')
    ax3.set_ylim(-3.5, 15.5)    


    ax1.plot(time, V, 'r', label='6.3$^o$ C')
    ax1.plot(time, V_, 'b', label=f'{temperature}$^o$ C')
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

def plot_lif(time, v, current, current_list, frequency_list, last_current):

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
        
        # Highlight current point using sorted data
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
        # Empty F-I plot when no data
        ax2.set_xlabel('Injected Current (nA)', fontsize=10)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=10)
        ax2.set_title('Current-Frequency (F-I) Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_gate_dynamics(time, m, h, n, p=None, q=None, g_M=0, g_Ca=0):
    """
    Plot the dynamics of gate variables
    
    Parameters:
    -----------
    time : array-like
        Time array (ms)
    m, h, n : array-like
        Standard HH gate variables
    p, q : array-like, optional
        Additional gate variables for M-type K+ and Ca2+
    g_M, g_Ca : float
        Conductances for optional channels
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot standard HH gates
    ax.plot(time, m, 'r', label='m (Na⁺ activation)')
    ax.plot(time, h, 'b', label='h (Na⁺ inactivation)')
    ax.plot(time, n, 'g', label='n (K⁺ activation)')
    
    # Plot additional gates if channels are active
    if g_M > 0 and p is not None:
        ax.plot(time, p, 'c', label='p (M-type K⁺)')
    if g_Ca > 0 and q is not None:
        ax.plot(time, q, 'm', label='q (Ca²⁺)')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Gate Variable Value')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Gate Dynamics')
    
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
    
    # V-n phase plot
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot(V, n, 'g')
    ax1.set_xlabel('Membrane Potential (mV)')
    ax1.set_ylabel('n (K⁺ activation)')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('K⁺ Activation Phase Plot')
    plt.tight_layout()
    figs.append(fig1)
    
    # V-m phase plot
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(V, m, 'r')
    ax2.set_xlabel('Membrane Potential (mV)')
    ax2.set_ylabel('m (Na⁺ activation)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Na⁺ Activation Phase Plot')
    plt.tight_layout()
    figs.append(fig2)
    
    # V-h phase plot
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.plot(V, h, 'b')
    ax3.set_xlabel('Membrane Potential (mV)')
    ax3.set_ylabel('h (Na⁺ inactivation)')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Na⁺ Inactivation Phase Plot')
    plt.tight_layout()
    figs.append(fig3)
    
    # V-dV/dt phase plot (phase plane)
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

def response_time(time, Vd, Vs, current_amplitude=None, current_input_start_time=None, excitatory_input_start_time=None, excitatory_input_strength=None, threshold=-20):
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
    Vs = Vs.tolist()
    Vd = Vd.tolist()

    if current_amplitude is not None:
        if current_amplitude > 0:

            if max(Vd) > threshold and max(Vs) > threshold:
                rise_time = np.round(time[next(x[0] for x in enumerate(Vs) if x[1] > threshold)] - current_input_start_time, 3)
                return rise_time
            elif max(Vd) < threshold and max(Vs) > threshold:
                rise_time = np.round(time[next(x[0] for x in enumerate(Vs) if x[1] > threshold)] - current_input_start_time, 3)
                return rise_time
            elif max(Vd) < threshold and max(Vs) < threshold:
                rise_time = np.round(time[Vs.index(max(Vs))] - current_input_start_time, 3)
                return rise_time
       #else:
        #    return None
    
    elif excitatory_input_strength is not None:
        if excitatory_input_strength > 0:
            if max(Vd) > threshold and max(Vs) > threshold:
                rise_time = np.round(time[next(x[0] for x in enumerate(Vs) if x[1] > threshold)] - excitatory_input_start_time, 3)
                return rise_time
            elif max(Vd) < threshold and max(Vs) > threshold:
                rise_time = np.round(time[next(x[0] for x in enumerate(Vs) if x[1] > threshold)] - excitatory_input_start_time, 3)
                return rise_time
            elif max(Vd) < threshold and max(Vs) < threshold:
                rise_time = np.round(time[Vs.index(max(Vs))] - excitatory_input_start_time, 3)
                return rise_time

    #else:
     #   return None
    
def analyze_spikes(time, V, threshold=-20):
    """
    Analyze spike properties and create plots
    
    Parameters:
    -----------
    time : array-like
        Time array (ms)
    V : array-like
        Membrane potential array (mV)
    threshold : float
        Threshold for spike detection (mV)
        
    Returns:
    --------
    results : dict
        Dictionary with spike analysis results
    figs : list
        List of figure objects
    """
    dt = time[1] - time[0]
    t_max = time[-1]
    
    # Find spike times
    spike_indices = np.where((V[:-1] < threshold) & (V[1:] >= threshold))[0]
    spike_times = time[spike_indices]
    
    results = {
        'n_spikes': len(spike_times),
        'spike_times': spike_times,
        'spike_indices': spike_indices
    }
    
    figs = []
    
    # Create spike raster plot if spikes detected
    if len(spike_times) > 0:
        # Raster plot
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.eventplot(spike_times, lineoffsets=0, linelengths=0.5, colors='k')
        ax1.set_xlabel('Time (ms)')
        ax1.set_yticks([])
        ax1.set_title('Spike Raster')
        plt.tight_layout()
        figs.append(fig1)
        
        # Calculate firing rate
        firing_rate = len(spike_times) / (t_max / 1000)  # in Hz
        results['firing_rate'] = firing_rate
        
        # Calculate interspike intervals if more than one spike
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            avg_isi = np.mean(isis)
            cv_isi = np.std(isis) / avg_isi if avg_isi > 0 else 0
            
            results['isis'] = isis
            results['avg_isi'] = avg_isi
            results['cv_isi'] = cv_isi
    
        # Plot first spike waveform if spikes detected
        if len(spike_indices) > 0:
            first_spike_idx = spike_indices[0]
            start_idx = max(0, first_spike_idx - int(5/dt))
            end_idx = min(len(time), first_spike_idx + int(15/dt))
            
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(time[start_idx:end_idx], V[start_idx:end_idx], 'b')
            ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Membrane Potential (mV)')
            ax2.set_title('First Spike Waveform')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            figs.append(fig2)
            
            # Calculate spike properties
            amplitude = np.max(V[first_spike_idx:end_idx]) - V[first_spike_idx]
            results['spike_amplitude'] = amplitude
            
            # Find spike width at half-maximum
            spike_peak = np.max(V[first_spike_idx:end_idx])
            half_max = (spike_peak + V[first_spike_idx]) / 2
            above_half_max = np.where(V[first_spike_idx:end_idx] >= half_max)[0]
            if len(above_half_max) > 1:
                width = (above_half_max[-1] - above_half_max[0]) * dt
                results['spike_width'] = width
    
    return results, figs
