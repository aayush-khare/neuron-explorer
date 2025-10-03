import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_somatic_membrane_potential_with_spike_counts(time, vs_control, vs_alt, temperature, input_current, input_list, control_data_list, alt_data_list, last_input):
    """
    Plot somatic membrane potential for HVC(RA) neuron with spike counts for two experimental 
    conditions (control condition 40 degree Celsius and an alternative condition for a different temperature)

    Parameters
    ----------
    time : array_like
        Time vector for the membrane potential traces.
    vs_control : array_like
        Somatic membrane potential values for the control condition.
    vs_alt : array_like
        Somatic membrane potential values for the alternative condition.
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.
    input_current : array_like
        Input current values
    input_list : list
        List of input current values applied during the experiment.
    control_data_list : list
        Spike count data corresponding to each input current for the control condition.
    alt_data_list : list
        Spike count data corresponding to each input current for the alternative condition.
    last_input : float
        The most recent input current value that was applied.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing the membrane potential traces
        and spike count plot for both experimental conditions.

    """

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax2 = fig.add_subplot(gs[:, 1])

    ax1.plot(time, vs_control, 'r', label='40.0$^o$ C')
    ax1.plot(time, vs_alt, 'b', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_xlim(100, 200)
    ax1.set_ylim(-100, 100)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Membrane Potential for the somatic compartment')

    ax3.plot(time, input_current, 'tab:green', linestyle='--')
    ax3.set_ylabel(f'input current ($\\mu A/cm^{2}$)', color='tab:green', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.set_ylim(-0.3, 1.1)
    ax3.set_xlim(100, 200)
    ax3.set_xlabel('Time (ms)', fontsize=12)

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
                    label=f'{last_input:.1f} $\\mu A/cm^{2}$, {current_control_data} spikes')
            ax2.plot(last_input, current_alt_data, 'o', color='blue', markersize=10,
                    label=f'{last_input:.1f} $\\mu A/cm^{2}$, {current_alt_data} spikes')
        
        ax2.set_xlabel(f'input current ($\\mu A/cm^{2}$)', fontsize=12)
        ax2.set_ylabel('num spikes', fontsize=12)
        ax2.set_title('Somatic spikes vs current injected', fontsize=14)
        ax2.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_ylim(-0.5, 5.5)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
    else:
        ax2.set_xlabel(f'input current ($\\mu A/cm^{2}$)', fontsize=12)
        ax2.set_ylabel('num spikes', fontsize=12)
        ax2.set_title('Somatic spikes vs current injected', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_somatic_membrane_potential_q10(time, vs_control, vs_alt, temperature, input_synapse, input_synapse_alt, q_cond, synaptic_input_list, response_time_control_list, responase_time_alt_list, last_synaptic_input):
    """
    Plot somatic membrane potential for HVC(RA) neuron and Q10 analysis for two temperature conditions.

    This function creates visualizations comparing membrane potential responses, a
    single synaptic input's profile and their temperature sensitivity (Q10) between 
    a control condition (40 degree Celsius) and an alternative temperature condition, 
    analyzing response times across different synaptic input strengths. 

    Parameters
    ----------
    time : array_like
        Time vector for the membrane potential traces.
    vs_control : array_like
        Somatic membrane potential values for the control condition.
    vs_alt : array_like
        Somatic membrane potential values for the alternative temperature condition.
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.
    input_synapse " array_like
        Input synapse values' array for the control condition
    input_synapse_alt : array_like
        Input synapse values' array for the alternative condition
    q_cond : float
        Q10 value set for diffusion dependent processes
    synaptic_input_list : list
        List of synaptic input strength values applied during the experiment.
    response_time_control_list : list
        Response time measurements corresponding to each synaptic input 
        for the control condition.
    response_time_alt_list : list
        Response time measurements corresponding to each synaptic input 
        for the alternative temperature condition.
    last_synaptic_input : float
        The most recent synaptic input strength value that was applied.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing membrane potential traces, synpatic
        input trace and Q10 analysis plots comparing both temperature conditions.

    """

    input_synapse_alt = [g / q_cond for g in input_synapse_alt]
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax2 = fig.add_subplot(gs[:, 1])

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

    ax3.plot(time, input_synapse, 'red', label='40.0$^o$ C', linestyle='-')
    ax3.plot(time, input_synapse_alt, 'blue', label=f'{temperature}$^o$ C', linestyle='-')
    ax3.set_ylabel(f'synaptic input ($mS/cm^{2}$)', fontsize=12)
    ax3.tick_params(axis='y')
    ax3.set_ylim(-0.3, 1.1)
    ax3.set_xlim(100, 200)
    ax3.set_xlabel('Time (ms)', fontsize=12)

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
    Plot the somatic membrane potential for the HVC(RA) neuron under a control 
    temperature (40 degree celsius) and an alternative temperature condition.

    Parameters
    ----------
    time : array_like
        Time vector for the membrane potential traces.
    vs_control : array_like
        Somatic membrane potential values for the control condition.
    vs_alt : array_like
        Somatic membrane potential values for the alternative condition.
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing the somatic membrane potential traces for both conditions.
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
    Plot the dendritic membrane potential for the HVC(RA) neuron under a control 
    temperature (40 degree celsius) and an alternative temperature condition.

    Parameters
    ----------
    time : array_like
        Time vector for the membrane potential traces.
    vd_control : array_like
        Dendritic membrane potential values for the control condition.
    vd_alt : array_like
        Dendritic membrane potential values for the alternative condition.
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing the dendritic membrane potential traces for both conditions.
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

def plot_hvci_membrane_potential(time, v_control, v_alt, input_profile, temperature, input_strength_list, frequency_list_control, frequency_list_alt, last_input_strength, input_type):

    """
    Plot the membrane potential for the HVC(I) neuron under a control 
    temperature (40 degree celsius) and an alternative temperature condition.

    Parameters
    ----------
    time : array_like
        Time vector for the membrane potential traces.
    v_control : array_like
        Membrane potential values for the control condition.
    v_alt : array_like
        Membrane potential values for the alternative condition.
    input_profile : array_like
        Vector containing values of the input made over the simulation time (either current or synaptic depending on the input type)
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.
    input_strength_list : array_like
        List of input strength values applied during the experiment (either current pulse strength or the maximal synaptic kick strength)
    frequency_list_control : array_like
        List of firing rate values calculated for all the applied input strengths so far under the control condition
    frequency_list_alt: array_like
        List of firing rate values calculated for all the applied input strengths so far under the alternative condition
    last_input_strength : float
        The most recent input strength value that was applied.
    input_type : str
        Either a 'Single current pulse' or 'Synaptic input in multiple intervals'
    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing plots for the membrane potential traces 
        and the firing rate vs maximum input strength applied for both conditions. 
    """

    if input_type == 'Single current pulse':
        x_label = 'Injected Current ($\\mu A/cm^{2}$)'
        y_label = f'current stimulus ($\\mu A/cm^{2}$)'
        plot_title = 'Membrane Potential and input profile'
        input_strength = 'current amplitude'
        input_units = f'$\\mu A/cm^{2}$'
        xmax = 300   
        ymax = 20.5
        y_ticks_list = [0.0, 5.0, 10.0, 15.0, 20.0,]
        linestyle = '--'
    
    elif input_type == 'Synaptic input in multiple intervals':
        x_label = 'max synaptic kick($mS/cm^{2}$)'
        y_label = f'synapse stimulus ($mS/cm^{2}$)'        
        plot_title = 'Membrane Potential and input profile'
        input_strength = 'input max kick'
        input_units = f'$mS/cm^{2}$'
        xmax = 600
        ymax = 5.5
        y_ticks_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        linestyle = '-'

    xmin = 0
    ymin = -1

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax2 = fig.add_subplot(gs[:, 1])

    ax1.plot(time, v_alt, 'b', label=f'{temperature}$^o$ C')
    ax1.plot(time, v_control, 'r', label='40.0$^o$ C', alpha=0.5)
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax1.set_ylim(-90, 50)
    ax1.set_xlim(xmin, xmax)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(plot_title)    

    ax3.plot(time, input_profile, 'tab:green', linestyle=linestyle)
    ax3.set_ylabel(y_label, color='tab:green', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.set_ylim(ymin, ymax)
    ax3.set_xlim(xmin, xmax)
    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.set_yticks(y_ticks_list)

    if len(input_strength_list) > 0:
        sorted_pairs = sorted(zip(input_strength_list, frequency_list_control, frequency_list_alt))
        sorted_current, sorted_frequency_control, sorted_frequency_alt = zip(*sorted_pairs)
        sorted_current = list(sorted_current)
        sorted_frequency_control = list(sorted_frequency_control)
        sorted_frequency_alt = list(sorted_frequency_alt)
        
        ax2.plot(sorted_current, sorted_frequency_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7)
        ax2.plot(sorted_current, sorted_frequency_alt,
                 'bo-', linewidth=1, markersize=4, alpha=0.7)
        
        if last_input_strength in [c for c in sorted_current if c == last_input_strength]:
            idx = next(i for i, c in enumerate(sorted_current) if c == last_input_strength)
            current_freq_control = sorted_frequency_control[idx]
            current_freq_alt = sorted_frequency_alt[idx]
            ax2.plot(last_input_strength, current_freq_control, 'o', color='red', markersize=10,
                    label=f'{last_input_strength:.1f} {input_units}, {current_freq_control:.1f} Hz')
            ax2.plot(last_input_strength, current_freq_alt, 'o', color='blue', markersize=10,
                    label=f'{last_input_strength:.1f} {input_units}, {current_freq_alt:.1f} Hz')   
        
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=12)
        ax2.set_title('Input-Frequency Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
      
    else:
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=12)
        ax2.set_title('Input-Frequency Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, 0.5, f'No data points yet\nAdjust {input_strength} and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)

    plt.tight_layout()
    return fig

def plot_hh_membrane_potential(time, v_control, v_alt, current, temperature, current_list, frequency_list_control, frequency_list_alt, last_current):
    
    """
    Plot the membrane potential for the Hodgkin Huxley neuron model under a control 
    temperature (6.3 degree celsius) and an alternative temperature condition.

    Parameters
    ----------
    time : array_like
        Time vector for the membrane potential traces.
    v_control : array_like
        Membrane potential values for the control condition.
    v_alt : array_like
        Membrane potential values for the alternative condition.
    current : array_like
        Vector containing values of the current input made over the simulation time
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.
    current_list : array_like
        List of current input strength values applied during the experiment
    frequency_list_control : array_like
        List of firing rate values calculated for all the applied current strengths so far under the control condition
    frequency_list_alt: array_like
        List of firing rate values calculated for all the applied current strengths so far under the alternative condition
    last_current : float
        The most recent current input value that was applied.
    
    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing plots for the membrane potential traces 
        and the firing rate vs current input strength applied for both conditions. 
    """
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax2 = fig.add_subplot(gs[:, 1])

    ax1.plot(time, v_control, 'r', label='6.3$^o$ C')
    ax1.plot(time, v_alt, 'b', label=f'{temperature}$^o$ C')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax1.set_ylim(-90, 50)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Membrane Potential and Stimulus Current')

    ax3.plot(time, current, 'tab:green', linestyle='--')
    ax3.set_ylabel(f'input current ($\\mu A/cm^{2}$)', color='tab:green', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.set_ylim(-2, 16)
    ax3.set_xlabel('Time (ms)', fontsize=12)

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
                    label=f'{last_current:.1f} $\\mu A/cm^{2}$, {current_freq_control:.1f} Hz')
            ax2.plot(last_current, current_freq_alt, 'o', color='blue', markersize=10,
                    label=f'{last_current:.1f} $\\mu A/cm^{2}$, {current_freq_alt:.1f} Hz')   
        
        ax2.set_xlabel('Injected Current ($\\mu A/cm^{2}$)', fontsize=12)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=12)
        ax2.set_title('Current-Frequency (F-I) Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
      
    else:
        ax2.set_xlabel('Injected Current ($\\mu A/cm^{2}$)', fontsize=12)
        ax2.set_ylabel('Firing Frequency (Hz)', fontsize=12)
        ax2.set_title('Current-Frequency (F-I) Relationship', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)


    plt.tight_layout()
    return fig

def plot_lif_membrane_potential(time, v, current, current_list, frequency_list, last_current):
    """
    Plot the membrane potential for the LIF neuron model.

    Parameters
    ----------
    time : array_like
        Time vector for the membrane potential trace.
    v : array_like
        Membrane potential values.
    current : array_like
        Vector containing values of the current input made over the simulation time
    current_list : array_like
        List of current input strength values applied during the experiment
    frequency_list : array_like
        List of firing rate values calculated for all the applied current strengths so far
    last_current : float
        The most recent current input value that was applied.
    
    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing plots for the membrane potential traces 
        and the firing rate vs current input strength applied. 
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

    ax3.plot(time, current, 'tab:green', linestyle='--')
    ax3.set_ylabel(f'current stimulus ($\\mu A/cm^{2})$', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')
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
                    label=f'{last_current:.2f} $\\mu A/cm^{2}$, {current_freq:.1f} Hz')
        
        ax2.set_xlabel(f'Injected Current ($\\mu A/cm^{2}$)', fontsize=10)
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