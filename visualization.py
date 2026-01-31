import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_somatic_membrane_potential_with_spike_counts(time, vs_control, vs_alt, temperature, input_current, input_list, control_data_list, alt_data_list, last_input):
    """
    Plot somatic membrane potential for HVC(RA) neuron with spike counts for two experimental 
    conditions (control condition for 40 degree Celsius and an alternative condition for a different temperature)

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
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time (ms)', fontsize=12)

    if len(input_list) > 0:

        ax2.plot(input_list, control_data_list,
                'ro-', linewidth=1, markersize=4, alpha=0.7)
        ax2.plot(input_list, alt_data_list,
                'bo-', linewidth=1, markersize=4, alpha=0.7)
        
        if last_input in [c for c in input_list if c == last_input]:
            idx = next(i for i, c in enumerate(input_list) if c == last_input)
            current_control_data = control_data_list[idx]
            current_alt_data = alt_data_list[idx]
            ax2.plot(last_input, current_control_data, 'o', color='red', markersize=10,
                    label=f'{last_input:.2f} $\\mu A/cm^{2}$, {current_control_data} spikes')
            ax2.plot(last_input, current_alt_data, 'o', color='blue', markersize=10,
                    label=f'{last_input:.2f} $\\mu A/cm^{2}$, {current_alt_data} spikes')
        
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
        Input synapse values' array for the control condition.
    input_synapse_alt : array_like
        Input synapse values' array for the alternative condition.
    q_cond : float
        Q10 value set for diffusion dependent processes.
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
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time (ms)', fontsize=12)

    if len(synaptic_input_list) > 0:

        response_q = [(r_ / r) ** (0.1 * (40.0 - temperature)) for r, r_ in zip(response_time_control_list, responase_time_alt_list)]

        ax2.plot(synaptic_input_list, response_q,
                'go-', linewidth=1, markersize=4, alpha=0.7, label='Measured points')
        
        if last_synaptic_input in [c for c in synaptic_input_list if c == last_synaptic_input]:
            idx = next(i for i, c in enumerate(synaptic_input_list) if c == last_synaptic_input)
            current_q10 = response_q[idx]
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

def plot_hvci_membrane_potential(time, v_control, v_alt, input_profile_control, input_profile_alt, q_cond, temperature, input_strength_list, frequency_list_control, frequency_list_alt, last_input_strength, input_type):

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
    input_profile_control : array_like
        Vector containing values of the input made over the simulation time (either current or synaptic depending on the input type), for control condition.
    input_profile_alt : array_like
        Vector containing values of the input made over the simulation time (either current or synaptic depending on the input type), for alternative condition.
    q_cond : float
        Q10 value set for diffusion dependent processes.
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.
    input_strength_list : array_like
        List of input strength values applied during the experiment (either current pulse strength or the maximal synaptic kick strength).
    frequency_list_control : array_like
        List of firing rate values calculated for all the applied input strengths so far under the control condition.
    frequency_list_alt: array_like
        List of firing rate values calculated for all the applied input strengths so far under the alternative condition.
    last_input_strength : float
        The most recent input strength value that was applied.
    input_type : str
        Either a 'Single current pulse' or 'Synaptic input in multiple intervals'.
    
    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing plots for the membrane potential traces 
        and the firing rate vs maximum input strength applied for both conditions. 
    """

    if input_type == 'Single current pulse':
        x_label = 'Injected Current ($\\mu A/cm^{2}$)'
        y_label = f'current stimulus ($\\mu A/cm^{2}$)'
        plot_title = 'Membrane Potential'
        input_strength = 'current amplitude'
        input_units = f'$\\mu A/cm^{2}$'
        input_title = 'Current input profile'
        xmax = 300   
        ymax = 20.5
        y_ticks_list = [0.0, 5.0, 10.0, 15.0, 20.0,]
        linestyle = '--'
    
    elif input_type == 'Synaptic input in multiple intervals':
        input_profile_alt = [g / q_cond for g in input_profile_alt]
        x_label = 'max synaptic kick($mS/cm^{2}$)'
        y_label = f'synapse stimulus ($mS/cm^{2}$)'        
        plot_title = 'Membrane Potential'
        input_strength = 'input max kick'
        input_units = f'$mS/cm^{2}$'
        input_title = 'Synaptic input profile'
        xmax = 1000
        ymax = 5.5
        y_ticks_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        linestyle = '-'

    xmin = 0
    ymin = -1

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax2 = fig.add_subplot(gs[:, 1])

    ax1.plot(time, v_control, 'r')
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax1.set_ylim(-90, 50)
    ax1.set_xlim(xmin, xmax)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(plot_title + ' (40.0$^o$ C)') 

    ax3.plot(time, v_alt, 'b')
    ax3.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax3.set_ylim(-90, 50)
    ax3.set_xlim(xmin, xmax)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(plot_title + f' ({temperature}$^o$ C)')    

    if input_type == 'Single current pulse':
        ax4.plot(time, input_profile_control, 'tab:green', linestyle=linestyle)
        ax4.set_ylabel(y_label, color='tab:green', fontsize=12)
        ax4.tick_params(axis='y', labelcolor='tab:green')
    else:
        ax4.plot(time, input_profile_control, 'red', linestyle=linestyle)
        ax4.plot(time, input_profile_alt, color='blue', linestyle=linestyle)
        ax4.set_ylabel(y_label, color='black', fontsize=12) 
        ax4.tick_params(axis='y', labelcolor='black')
    ax4.set_ylim(ymin, ymax)
    ax4.set_xlim(xmin, xmax)
    ax4.set_xlabel('Time (ms)', fontsize=12)
    ax4.set_yticks(y_ticks_list)
    ax4.set_title(input_title)
    ax4.grid(True, alpha=0.3)

    if len(input_strength_list) > 0:

        ax2.plot(input_strength_list, frequency_list_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7)
        ax2.plot(input_strength_list, frequency_list_alt,
                 'bo-', linewidth=1, markersize=4, alpha=0.7)
        
        if last_input_strength in [c for c in input_strength_list if c == last_input_strength]:
            idx = next(i for i, c in enumerate(input_strength_list) if c == last_input_strength)
            current_freq_control = frequency_list_control[idx]
            current_freq_alt = frequency_list_alt[idx]
            ax2.plot(last_input_strength, current_freq_control, 'o', color='red', markersize=10,
                    label=f'{last_input_strength:.2f} {input_units}, {current_freq_control:.2f} Hz')
            ax2.plot(last_input_strength, current_freq_alt, 'o', color='blue', markersize=10,
                    label=f'{last_input_strength:.2f} {input_units}, {current_freq_alt:.2f} Hz')   
        
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
        Vector containing values of the current input made over the simulation time.
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.
    current_list : array_like
        List of current input strength values applied during the experiment.
    frequency_list_control : array_like
        List of firing rate values calculated for all the applied current strengths so far under the control condition.
    frequency_list_alt: array_like
        List of firing rate values calculated for all the applied current strengths so far under the alternative condition.
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
    ax1.set_title('Membrane Potential')

    ax3.plot(time, current, 'tab:green', linestyle='--')
    ax3.set_ylabel(f'input current ($\\mu A/cm^{2}$)', color='tab:green', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.set_ylim(-2, 16)
    ax3.set_title('Stimulus Current')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time (ms)', fontsize=12)

    if len(current_list) > 0:

        ax2.plot(current_list, frequency_list_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7)
        ax2.plot(current_list, frequency_list_alt,
                 'bo-', linewidth=1, markersize=4, alpha=0.7)
        
        if last_current in [c for c in current_list if c == last_current]:
            idx = next(i for i, c in enumerate(current_list) if c == last_current)
            current_freq_control = frequency_list_control[idx]
            current_freq_alt = frequency_list_alt[idx]
            ax2.plot(last_current, current_freq_control, 'o', color='red', markersize=10,
                    label=f'{last_current:.2f} $\\mu A/cm^{2}$, {current_freq_control:.2f} Hz')
            ax2.plot(last_current, current_freq_alt, 'o', color='blue', markersize=10,
                    label=f'{last_current:.2f} $\\mu A/cm^{2}$, {current_freq_alt:.2f} Hz')   
        
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

def plot_hvcra_somatic_spike_characteristics(temperature, input_list, input_type, spike_width_list_control, spike_width_list_alt, isi_list_control, isi_list_alt, last_input):
    """
    Plot the mean spike width (ms) and Inter-Spike Intervals (ISI, ms) as input strength is varied
    at the control and alternative temperature conditions for the HVC(RA) neuron model.

    Parameters
    ----------
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.
    input_list : array_like
        List of input strength values applied during the experiment (either current pulse strength or the maximal synaptic kick strength).
    input_type : str
        Either 'Current input' or 'Synaptic input'.
    spike_width_list_control : array_like
        List of mean spike width values obtained as input strength is varied for the control condition.
    spike_width_list_alt : array_like
        List of mean spike width values obtained as input strength is varied for the alternative condition.
    isi_list_control : array_like
        List of ISI values obtained as input strength is varied for the control condition.
    isi_list_alt : array_like
        List of ISI values obtained as input strength is varied for the alternative condition.
    last_input : float
        The most recent input strength value that was applied.
    
    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing plots for the mean spike width and ISI as
        input strength is varied at the two temperature conditions.
    """

    if input_type == 'Current input':
        input_units = '$\\mu A/cm^{2}$'
        x_label = 'Injected Current ($\\mu A/cm^{2}$)'
    elif input_type == 'Synaptic input':
        input_units = '$mS/cm^{2}$'
        x_label = 'Excitatory synaptic input, gE (mS/cm$^2$)'


    fig = plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    
    if len(input_list) > 0:

        mask = np.array(spike_width_list_control) != 0 
        filtered_current = np.array(input_list)[mask]
        filtered_width_control = np.array(spike_width_list_control)[mask]
        ax1.plot(filtered_current, filtered_width_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7, label = '40.0$^o$ C')
        
        mask = np.array(spike_width_list_alt) != 0 
        filtered_current = np.array(input_list)[mask]
        filtered_width_alt = np.array(spike_width_list_alt)[mask]
        ax1.plot(filtered_current, filtered_width_alt,
                'bo-', linewidth=1, markersize=4, alpha=0.7, label = f'{temperature}$^o$ C')
        ax1.set_ylim(0, 1.0)

        mask = np.array(isi_list_control) != 0 
        filtered_current = np.array(input_list)[mask]
        filtered_isi_control = np.array(isi_list_control)[mask]
        ax2.plot(filtered_current, filtered_isi_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7)
        
        mask = np.array(isi_list_alt) != 0 
        filtered_current = np.array(input_list)[mask]
        filtered_isi_alt = np.array(isi_list_alt)[mask]
        ax2.plot(filtered_current, filtered_isi_alt,
                'bo-', linewidth=1, markersize=4, alpha=0.7)
        ax2.set_ylim(0, 10.0)
        
        if last_input in [c for c in input_list if c == last_input]:
            idx = next(i for i, c in enumerate(input_list) if c == last_input)
            current_spike_width_control = spike_width_list_control[idx]
            current_spike_width_alt = spike_width_list_alt[idx]
            current_isi_control = isi_list_control[idx]
            current_isi_alt = isi_list_alt[idx]

            if current_spike_width_control != 0:
                ax1.plot(last_input, current_spike_width_control, 'o', color='red', markersize=10,
                        label=f'{last_input:.2f} {input_units}, {current_spike_width_control:.2f} ms')
            if current_spike_width_alt != 0:
                ax1.plot(last_input, current_spike_width_alt, 'o', color='blue', markersize=10,
                        label=f'{last_input:.2f} {input_units}, {current_spike_width_alt:.2f} ms')   
            if current_isi_control != 0:
                ax2.plot(last_input, current_isi_control, 'o', color='red', markersize=10,
                        label=f'{last_input:.2f} {input_units}, {current_isi_control:.2f} ms')
            if current_isi_alt != 0:
                ax2.plot(last_input, current_isi_alt, 'o', color='blue', markersize=10,
                        label=f'{last_input:.2f} {input_units}, {current_isi_alt:.2f} ms')   
        
        ax1.set_ylabel('Mean spike width (ms)', fontsize=6)
        ax2.set_ylabel('ISI (ms)', fontsize=6)
        ax2.set_xlabel(x_label, fontsize=6)
        ax1.set_title('Change in spike widths and Interspike intervals (ISIs) with temperature', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.tick_params(axis='both', labelsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.grid(True, alpha=0.3)
        if current_isi_alt != 0 or current_isi_control != 0:
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))        
      
    else:
        ax1.set_ylabel('Mean spike width (ms)', fontsize=6)
        ax2.set_xlabel(x_label, fontsize=6)
        ax2.set_ylabel('Interspike interval (ms)', fontsize=6)
        ax1.set_title('Change in spike widths and Interspike intervals (ISIs) with temperature', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        ax1.text(0.5, 0.5, 'No data points yet\nAdjust input strength and observe', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax2.text(0.5, 0.5, 'No data points yet\nAdjust input strength and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
    plt.tight_layout()
    return fig

def plot_hvci_spike_characteristics(temperature, input_strength_list, spike_width_list_control, spike_width_list_alt, isi_list_control, isi_list_alt, last_input_strength):
    """
    Plot the mean spike width (ms) and Inter-Spike Intervals (ISI, ms) as input strength is varied
    at the control and alternative temperature conditions for the HVC(I) neuron model.

    Parameters
    ----------
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.
    input_strength_list : array_like
        List of input strength values applied during the experiment.
    spike_width_list_control : array_like
        List of mean spike width values obtained as input strength is varied for the control condition.
    spike_width_list_alt : array_like
        List of mean spike width values obtained as input strength is varied for the alternative condition.
    isi_list_control : array_like
        List of ISI values obtained as input strength is varied for the control condition.
    isi_list_alt : array_like
        List of ISI values obtained as input strength is varied for the alternative condition.
    last_input_strength : float
        The most recent input strength value that was applied.
    
    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing plots for the mean spike width and ISI as
        input strength is varied at the two temperature conditions.
    """
    fig = plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    
    if len(input_strength_list) > 0:

        mask = np.array(spike_width_list_control) != 0 
        filtered_current = np.array(input_strength_list)[mask]
        filtered_width_control = np.array(spike_width_list_control)[mask]
        ax1.plot(filtered_current, filtered_width_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7, label = '40.0$^o$ C')
        
        mask = np.array(spike_width_list_alt) != 0 
        filtered_current = np.array(input_strength_list)[mask]
        filtered_width_alt = np.array(spike_width_list_alt)[mask]
        ax1.plot(filtered_current, filtered_width_alt,
                'bo-', linewidth=1, markersize=4, alpha=0.7, label = f'{temperature}$^o$ C')
        ax1.set_ylim(0, 1.0)

        mask = np.array(isi_list_control) != 0 
        filtered_current = np.array(input_strength_list)[mask]
        filtered_isi_control = np.array(isi_list_control)[mask]
        ax2.plot(filtered_current, filtered_isi_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7)
        
        mask = np.array(isi_list_alt) != 0 
        filtered_current = np.array(input_strength_list)[mask]
        filtered_isi_alt = np.array(isi_list_alt)[mask]
        ax2.plot(filtered_current, filtered_isi_alt,
                'bo-', linewidth=1, markersize=4, alpha=0.7)
        ax2.set_ylim(0, 30.0)
        
        if last_input_strength in [c for c in input_strength_list if c == last_input_strength]:
            idx = next(i for i, c in enumerate(input_strength_list) if c == last_input_strength)
            current_spike_width_control = spike_width_list_control[idx]
            current_spike_width_alt = spike_width_list_alt[idx]
            current_isi_control = isi_list_control[idx]
            current_isi_alt = isi_list_alt[idx]

            if current_spike_width_control != 0:
                ax1.plot(last_input_strength, current_spike_width_control, 'o', color='red', markersize=10,
                        label=f'{last_input_strength:.2f} $\\mu A/cm^{2}$, {current_spike_width_control:.2f} ms')
            if current_spike_width_alt != 0:
                ax1.plot(last_input_strength, current_spike_width_alt, 'o', color='blue', markersize=10,
                        label=f'{last_input_strength:.2f} $\\mu A/cm^{2}$, {current_spike_width_alt:.2f} ms')   
            if current_isi_control != 0:
                ax2.plot(last_input_strength, current_isi_control, 'o', color='red', markersize=10,
                        label=f'{last_input_strength:.2f} $\\mu A/cm^{2}$, {current_isi_control:.2f} ms')
            if current_isi_alt != 0:
                ax2.plot(last_input_strength, current_isi_alt, 'o', color='blue', markersize=10,
                        label=f'{last_input_strength:.2f} $\\mu A/cm^{2}$, {current_isi_alt:.2f} ms')   
        
        ax1.set_ylabel('Mean spike width (ms)', fontsize=6)
        ax2.set_ylabel('ISI (ms)', fontsize=6)
        ax2.set_xlabel('Injected Current ($\\mu A/cm^{2}$)', fontsize=6)
        ax1.set_title('Change in spike widths and Interspike intervals (ISIs) with temperature', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.tick_params(axis='both', labelsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.grid(True, alpha=0.3)
        
        if current_isi_alt != 0 or current_isi_control != 0:
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))        
      
    else:
        ax1.set_ylabel('Mean spike width (ms)', fontsize=6)
        ax2.set_xlabel('Injected Current ($\\mu A/cm^{2}$)', fontsize=6)
        ax2.set_ylabel('Interspike interval (ms)', fontsize=6)
        ax1.set_title('Change in spike widths and Interspike intervals (ISIs) with temperature', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        ax1.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax2.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
    plt.tight_layout()
    return fig

def plot_hh_membrane_potential_spike_characteristics(temperature, current_list, spike_width_list_control, spike_width_list_alt, isi_list_control, isi_list_alt, last_current):
    """
    Plot the mean spike width (ms) and Inter-Spike Intervals (ISI, ms) as input strength is varied
    at the control and alternative temperature conditions for the Hodgkin Huxley (HH) neuron model.

    Parameters
    ----------
    temperature : float
        Temperature (in °C) used for the alternative experimental condition.
    current_list : array_like
        List of current input values applied during the experiment.
    spike_width_list_control : array_like
        List of mean spike width values obtained as input strength is varied for the control condition.
    spike_width_list_alt : array_like
        List of mean spike width values obtained as input strength is varied for the alternative condition.
    isi_list_control : array_like
        List of ISI values obtained as input strength is varied for the control condition.
    isi_list_alt : array_like
        List of ISI values obtained as input strength is varied for the alternative condition.
    last_current : float
        The most recent current input value that was applied.
    
    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing plots for the mean spike width and ISI as
        current strength is varied at the two temperature conditions.
    """

    fig = plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    
    if len(current_list) > 0:
        
        mask = np.array(spike_width_list_control) != 0 
        filtered_current = np.array(current_list)[mask]
        filtered_width_control = np.array(spike_width_list_control)[mask]
        ax1.plot(filtered_current, filtered_width_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7, label = '6.3$^o$ C')
        
        mask = np.array(spike_width_list_alt) != 0 
        filtered_current = np.array(current_list)[mask]
        filtered_width_alt = np.array(spike_width_list_alt)[mask]
        ax1.plot(filtered_current, filtered_width_alt,
                'bo-', linewidth=1, markersize=4, alpha=0.7, label = f'{temperature}$^o$ C')
        ax1.set_ylim(0, 5.0)

        mask = np.array(isi_list_control) != 0 
        filtered_current = np.array(current_list)[mask]
        filtered_isi_control = np.array(isi_list_control)[mask]
        ax2.plot(filtered_current, filtered_isi_control,
                'ro-', linewidth=1, markersize=4, alpha=0.7)
        
        mask = np.array(isi_list_alt) != 0 
        filtered_current = np.array(current_list)[mask]
        filtered_isi_alt = np.array(isi_list_alt)[mask]
        ax2.plot(filtered_current, filtered_isi_alt,
                'bo-', linewidth=1, markersize=4, alpha=0.7)
        ax2.set_ylim(0, 40.0)

        if last_current in [c for c in current_list if c == last_current]:
            idx = next(i for i, c in enumerate(current_list) if c == last_current)
            current_spike_width_control = spike_width_list_control[idx]
            current_spike_width_alt = spike_width_list_alt[idx]
            current_isi_control = isi_list_control[idx]
            current_isi_alt = isi_list_alt[idx]

            if current_spike_width_control != 0:
                ax1.plot(last_current, current_spike_width_control, 'o', color='red', markersize=10,
                        label=f'{last_current:.2f} $\\mu A/cm^{2}$, {current_spike_width_control:.2f} ms')
            if current_spike_width_alt != 0:
                ax1.plot(last_current, current_spike_width_alt, 'o', color='blue', markersize=10,
                        label=f'{last_current:.2f} $\\mu A/cm^{2}$, {current_spike_width_alt:.2f} ms')   
            if current_isi_control != 0:    
                ax2.plot(last_current, current_isi_control, 'o', color='red', markersize=10,
                        label=f'{last_current:.2f} $\\mu A/cm^{2}$, {current_isi_control:.2f} ms')
            if current_isi_alt != 0:    
                ax2.plot(last_current, current_isi_alt, 'o', color='blue', markersize=10,
                        label=f'{last_current:.2f} $\\mu A/cm^{2}$, {current_isi_alt:.2f} ms')   
        
        ax1.set_ylabel('Mean spike width (ms)', fontsize=6)
        ax2.set_ylabel('ISI (ms)', fontsize=6)
        ax2.set_xlabel('Injected Current ($\\mu A/cm^{2}$)', fontsize=6)
        ax1.set_title('Change in spike widths and Interspike intervals (ISIs) with temperature', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.tick_params(axis='both', labelsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.grid(True, alpha=0.3)
        if current_isi_alt != 0 or current_isi_control != 0:
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
      
    else:
        ax1.set_ylabel('Mean spike width (ms)', fontsize=6)
        ax2.set_xlabel('Injected Current ($\\mu A/cm^{2}$)', fontsize=6)
        ax2.set_ylabel('Interspike interval (ms)', fontsize=6)
        ax1.set_title('Change in spike widths and Interspike intervals (ISIs) with temperature', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        ax1.text(0.5, 0.5, 'No data points yet\nAdjust current and observe', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
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
        Vector containing values of the current input made over the simulation time.
    current_list : array_like
        List of current input strength values applied during the experiment.
    frequency_list : array_like
        List of firing rate values calculated for all the applied current strengths so far.
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
        
        ax2.plot(current_list, frequency_list, 
                'o-', color='black', linewidth=1, markersize=4, alpha=1, label='Measured points')
        
        if last_current in [c for c in current_list if c == last_current]:
            idx = next(i for i, c in enumerate(current_list) if c == last_current)
            current_freq = frequency_list[idx]
            ax2.plot(last_current, current_freq, 'ro', markersize=6,
                    label=f'{last_current:.2f} $\\mu A/cm^{2}$, {current_freq:.2f} Hz')
        
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