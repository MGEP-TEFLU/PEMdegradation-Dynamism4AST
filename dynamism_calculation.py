import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rainflow
import os

def process_voltage_file(file_path, plot=False):
    """
    Process an Excel file containing time-voltage data and calculate
    dynamism metrics using Rainflow cycle counting.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing two columns: [time, voltage].
    plot : bool, optional
        If True, plots the voltage profile. Default is False.

    Returns
    -------
    dynamism : list
        A list of five metrics:
        [mean_voltage, median_amplitude, median_period,
         median_positive_dynamism, median_negative_dynamism]
    """
    
    # --- Read data ---
    data = pd.read_excel(file_path).values
    time = data[:, 0]
    voltage = data[:, 1]

    # --- Optional plot of voltage vs time ---
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(time, voltage, 'b-', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Voltage profile: {file_path}')
        plt.grid(True)
        plt.show()

    # --- Apply Rainflow counting ---
    cycles = list(rainflow.extract_cycles(voltage))

    # Handle empty cycles (constant signals)
    if not cycles:
        return [np.mean(voltage), 0, 1e7, 0, 0]

    rf = []
    for amp, mean, count, idx_start, idx_end in cycles:
        t_start = time[int(idx_start)]
        t_end = time[int(idx_end)]
        period = abs(t_end - t_start) / count
        rf.append([amp, mean, count, t_start, t_end, period])
    rf = np.array(rf)

    col1 = np.abs(rf[:, 0])  # amplitude
    col6 = rf[:, 5]          # period

    extra_cols = np.zeros((rf.shape[0], 5))  # columns 7–11
    rf = np.hstack([rf, extra_cols])

    # --- Detect constant segments in the voltage signal ---
    results = []
    start = 0
    for i in range(1, len(voltage)):
        if voltage[i] != voltage[i - 1]:
            if i - 1 > start:
                results.append([time[start], time[i - 1], voltage[i - 1]])
            start = i
    if len(voltage) >= start + 1:
        results.append([time[start], time[-1], voltage[-1]])

    results = np.array(results)
    n_rf = rf.shape[0]
    n_res = results.shape[0]

    # Pad or trim results to match dimensions with rf
    if n_res < n_rf:
        padding = np.zeros((n_rf - n_res, 3))
        results = np.vstack([results, padding])
    elif n_res > n_rf:
        results = results[:n_rf, :]

    # Merge segment information into rf array
    for i in range(results.shape[0]):
        t_start = results[i, 0]
        idx_start = np.where(rf[:, 3] == t_start)[0]
        if idx_start.size > 0:
            rf[idx_start[0], 6:9] = results[i]
        t_end = results[i, 1]
        idx_end = np.where(rf[:, 4] == t_end)[0]
        if idx_end.size > 0:
            rf[idx_end[0], 6:9] = results[i]

    # --- Assign sign to amplitude based on voltage direction ---
    for i in range(rf.shape[0]):
        t_start = rf[i, 3]
        t_end = rf[i, 4]
        idx_start = np.argmin(np.abs(time - t_start))
        idx_end = np.argmin(np.abs(time - t_end))
        v_start = voltage[idx_start]
        v_end = voltage[idx_end]
        rf[i, 0] = np.sign(v_end - v_start) * abs(rf[i, 0])

    # --- Compute derived metrics ---
    rf[:, 9] = (rf[:, 7] - rf[:, 6]) / 0.5
    rf[:, 10] = np.where(
        rf[:, 9] == rf[:, 5],
        0,
        rf[:, 0] / (0.5 * (rf[:, 5] - rf[:, 9]))
    )

    # --- Compute overall signal dynamism metrics ---
    dynamism = [
        np.mean(voltage),                                   # mean voltage
        np.median(np.abs(rf[:, 0])),                        # median amplitude
        np.median(rf[:, 5]),                                # median period
        np.median(rf[rf[:, 0] >= 0][:, 10]) if np.any(rf[:, 0] >= 0) else 0,  # median positive dynamism
        np.median(rf[rf[:, 0] <= 0][:, 10]) if np.any(rf[:, 0] <= 0) else 0   # median negative dynamism
    ]
    return dynamism


# --- File list ---
file_paths = [
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\Hold.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\triangle.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\sawtooth_down.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\sawtooth_up.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\square_16V_60s.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\square_18V_60s.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\square_2V_60s.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\square_22V_60s.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\square_25V_60s.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\square_2V_30s.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\square_2V_20s.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\square_2V_10s.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\solar.xlsx',
    r'C:\Users\jaizpuru\Desktop\jon\ikerketa\experimental\Rainflow\wind.xlsx'
]

# --- Process all files ---
dynamism_matrix = np.array([process_voltage_file(fp) for fp in file_paths])

# Dynamism reference limits (note that this limits can be adapted to each case and each type of signals: current, voltage, power...)
limits = np.array([2, 1.05, 10, 0.05, -0.05])  # indices 0–4

# Copy to avoid modifying original
dynamism_with_score = dynamism_matrix.copy()

# Compute normalized divisions
divisions = dynamism_with_score / limits

# --- Special handling for column 0 (mean voltage) ---
current_vals_col0 = dynamism_with_score[:, 0]
max_val_col0 = np.nanmax(current_vals_col0)
divisions[:, 0] = (current_vals_col0 - 1.45) / (limits[1] - 1.45)
divisions[:, 0] = np.clip(divisions[:, 0], 0, 1)  # clamp between 0–1

# --- Special handling for column 2 (period) ---
divisions[:, 2] = limits[2] / dynamism_with_score[:, 2]

# Cap values greater than 1
divisions = np.minimum(divisions, 1)

# Sum across rows ignoring NaNs
column6 = np.nansum(divisions, axis=1)

# Add as new column (index 6)
dynamism_with_score = np.hstack([dynamism_with_score, column6[:, np.newaxis]])

# --- Experimental degradation data ---
degradation_05 = [28.3809524, 138.5714, 174.6667, 222.095, 22.8572, 84.7619,
         202.8571, 229.5238, 246.667, 249.5238, 279.4286, 369.5238,
         25.4629, 27.777]

degradation_1= [19.4285714, 137.5238, 180.381, 242.2857, 26.0952, 88.7619, 
          215.619, 261.7142, 281.1428, 255.619, 293.5238, 380.5714,
          26.0185, 27.639]

degradation_175= [23.4285714, 154.2857, 208.5714, 284, 21.9048, 101.9048,
          256.19, 322.8571, 343.8095, 267.619, 307.619, 388, 25.6944, 28.472]

signal_type = [
    'Hold', 'Triangle', 'Sawtooth down', 'Sawtooth up',
    'Square', 'Square', 'Square', 'Square',
    'Square', 'Square', 'Square', 'Square',
    'Solar', 'Wind'
]

# --- Dynamism scores (col 6 in MATLAB = col 5 here) ---
x_vals = dynamism_with_score[:, 5]

# --- Color map for each signal type ---
color_map = {
    'Hold': [0, 0, 1],
    'Triangle': [0, 1, 1],
    'Sawtooth down': [1, 0, 0],
    'Sawtooth up': [1, 0.5, 0],
    'Square': [0, 1, 0],
    'Solar': [0.5, 0, 0.5],
    'Wind': [0.5, 0.5, 0]
}

# --- Plot degradation vs dynamism ---
plt.figure(figsize=(10, 6))
unique_types = sorted(set(signal_type))

signal_type = np.array(signal_type)
degradation_05 = np.array(degradation_05)
degradation_1 = np.array(degradation_1)
degradation_175 = np.array(degradation_175)

for t in unique_types:
    idx = signal_type == t
    c = color_map[t]

    plt.scatter(x_vals[idx], degradation_05[idx], s=60, marker='D',
                edgecolors='k', facecolors=c, label=t)
    plt.scatter(x_vals[idx], degradation_1[idx], s=60, marker='o',
                edgecolors='k', facecolors=c)
    plt.scatter(x_vals[idx], degradation_175[idx], s=60, marker='s',
                edgecolors='k', facecolors=c)

plt.xlabel('Dynamism')
plt.ylabel('Degradation (μV/h)')
plt.title('Degradation vs Dynamism')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Polynomial fit (2nd order) ---
all_degradations = np.concatenate([degradation_05, degradation_1, degradation_175])
all_dynamisms = np.tile(x_vals, 3)

mask = ~np.isnan(all_degradations) & ~np.isnan(all_dynamisms)
x_clean = all_dynamisms[mask]
y_clean = all_degradations[mask]

coeffs = np.polyfit(x_clean, y_clean, 2)
poly_fit = np.poly1d(coeffs)

x_fit = np.linspace(np.min(x_clean), np.max(x_clean), 200)
y_fit = poly_fit(x_fit)

# Compute R²
y_pred = poly_fit(x_clean)
ss_res = np.sum((y_clean - y_pred)**2)
ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
r2 = 1 - (ss_res / ss_tot)

# --- Plot regression curve ---
plt.figure(figsize=(10, 6))

for t in unique_types:
    idx = signal_type == t
    c = color_map[t]
    plt.scatter(x_vals[idx], degradation_05[idx], s=60, marker='D',
                edgecolors='k', facecolors=c, label=t)
    plt.scatter(x_vals[idx], degradation_1[idx], s=60, marker='o',
                edgecolors='k', facecolors=c)
    plt.scatter(x_vals[idx], degradation_175[idx], s=60, marker='s',
                edgecolors='k', facecolors=c)

plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='2nd order polynomial regression')

plt.xlabel('Dynamism')
plt.ylabel('Degradation (μV/h)')
plt.title(f'Degradation vs Dynamism\n2nd order fit: '
          f'y = {coeffs[0]:.3e}x² + {coeffs[1]:.3e}x + {coeffs[2]:.3e} | R² = {r2:.3f}')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()

plt.show()

