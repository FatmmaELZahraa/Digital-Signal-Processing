import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# --- Task 1 Functions ---
def load_signal_from_file_task1():
    file_path = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])
    if file_path:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                signal_type = lines[0].strip().split()[0]
                num_samples = int(lines[2].strip())

                time_data = []
                amplitude_data = []

                for line in lines[3:3 + num_samples]:
                    parts = list(map(float, line.strip().split()))
                    time_data.append(parts[0])
                    amplitude_data.append(parts[1])

                ax1_task1.clear()
                ax2_task1.clear()

                ax1_task1.plot(time_data, amplitude_data, color='cyan', marker='o', markersize=5)
                ax1_task1.set_facecolor('black')
                ax1_task1.set_title('Loaded Signal from File - Continuous', color='white')
                ax1_task1.set_xlabel('Time (s)', color='white')
                ax1_task1.set_ylabel('Amplitude', color='white')
                ax1_task1.tick_params(colors='red')

                ax2_task1.stem(time_data, amplitude_data, basefmt=" ", linefmt="red", markerfmt="ro")
                ax2_task1.set_facecolor('black')
                ax2_task1.set_title('Loaded Signal from File - Discrete (Stem)', color='white')
                ax2_task1.set_xlabel('Time (s)', color='white')
                ax2_task1.set_ylabel('Amplitude', color='white')
                ax2_task1.tick_params(colors='red')

                canvas_task1.draw()
        except Exception as e:
            error_label.config(text=f"Error reading file: {str(e)}")


def task1_update_plot():
    try:
        amplitude = float(amp_entry.get())
        sampling_frequency = float(freq_samp_entry.get())
        analog_frequency = float(freq_entry.get())
        phase_shift = float(phase_entry.get())
        signal_type = signal_menu.get()

        t = np.arange(0, 1, 1 / sampling_frequency)

        if signal_type == 'Sine':
            signal = amplitude * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
        else:
            signal = amplitude * np.cos(2 * np.pi * analog_frequency * t + phase_shift)

        ax1_task1.clear()
        ax2_task1.clear()

        ax1_task1.plot(t, signal)
        ax1_task1.set_facecolor('black')
        ax1_task1.set_title(f'{signal_type} Wave - Continuous', color='white')
        ax1_task1.set_xlabel('Time (s)', color='white')
        ax1_task1.set_ylabel('Amplitude', color='white')
        ax1_task1.tick_params(colors='red')

        ax2_task1.stem(t, signal, basefmt=" ", linefmt="red", markerfmt="ro")
        ax2_task1.set_facecolor('black')
        ax2_task1.set_title(f'{signal_type} Wave - Discrete (Stem)', color='white')
        ax2_task1.set_xlabel('Time (s)', color='white')
        ax2_task1.set_ylabel('Amplitude', color='white')
        ax2_task1.tick_params(colors='red')

        canvas_task1.draw()

    except ValueError:
        error_label.config(text="Please enter valid numeric values.")


# --- Task 2 Functions ---
def load_signal_from_file_task2(file_path):
    if file_path:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                num_samples = int(lines[2].strip())
                time_data = []
                amplitude_data = []

                for line in lines[3:3 + num_samples]:
                    parts = list(map(float, line.strip().split()))
                    time_data.append(parts[0])
                    amplitude_data.append(parts[1])

                return time_data, amplitude_data
        except Exception as e:
            error_label.config(text=f"Error reading file: {str(e)}")
            return None, None
    return None, None


def SignalSamplesAreEqual(file_name, indices, samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break

    if len(expected_samples) != len(samples):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")


def add_signals(signals):
    result = np.zeros_like(signals[0])
    for signal in signals:
        result += signal
    return result


def subtract_signals(signals):
    result = signals[0]
    for signal in signals[1:]:
        result -= np.abs(signal)
    return result


def multiply_signal(amplitudes, constant):
    return np.array([amplitude * constant for amplitude in amplitudes])


# def square_amplitudes(amplitudes):
#    return np.array(amplitudes) ** 2
def square_amplitudes(amplitudes):
    return np.square(amplitudes)


def normalize_signal(amplitudes, to_zero_one=True):
    min_val = np.min(amplitudes)
    max_val = np.max(amplitudes)
    if to_zero_one:
        return (amplitudes - min_val) / (max_val - min_val)
    else:
        return 2 * (amplitudes - min_val) / (max_val - min_val) - 1


def accumulation_amplitudes(amplitudes):
    cumulative = 0
    result = []
    for num in amplitudes:
        cumulative += num
        result.append(cumulative)
    return result


def task2_update_plot():
    operation = operation_menu.get()

    if operation in ["Addition", "Subtraction"]:
        file_paths1 = filedialog.askopenfilenames(title="Select First Set of Signal Files",
                                                  filetypes=[("Text Files", "*.txt")])
        if not file_paths1:
            return

        # Select the second set of signal files
        file_paths2 = filedialog.askopenfilenames(title="Select Second Set of Signal Files",
                                                  filetypes=[("Text Files", "*.txt")])
        if not file_paths2:
            return

        signals1 = []
        signals2 = []

        for file_path in file_paths1:
            time_data, amplitude_data = load_signal_from_file_task2(file_path)
            if time_data is None or amplitude_data is None:
                return
            signals1.append(np.array(amplitude_data))

        for file_path in file_paths2:
            time_data, amplitude_data = load_signal_from_file_task2(file_path)
            if time_data is None or amplitude_data is None:
                return
            signals2.append(np.array(amplitude_data))

        if operation == "Addition":
            result_amplitudes = add_signals(signals1) + add_signals(signals2)
        elif operation == "Subtraction":
            result_amplitudes = np.abs(subtract_signals(signals1) - add_signals(signals2))  # Taking absolute difference

        ax1_task2.clear()
        ax2_task2.clear()

        for signal in signals1:
            ax1_task2.plot(time_data, signal, color='cyan', marker='o', markersize=5)

        ax1_task2.set_facecolor('black')
        ax1_task2.set_title('Original Signals from First Set', color='white')
        ax1_task2.set_xlabel('Time (s)', color='white')
        ax1_task2.set_ylabel('Amplitude', color='white')
        ax1_task2.tick_params(colors='red')

        ax2_task2.plot(time_data, result_amplitudes, color='magenta', marker='x', markersize=5)
        ax2_task2.set_facecolor('black')
        ax2_task2.set_title(f'{operation} Result', color='white')
        ax2_task2.set_xlabel('Time (s)', color='white')
        ax2_task2.set_ylabel('Amplitude', color='white')
        ax2_task2.tick_params(colors='red')

        canvas_task2.draw()
    else:
        file_path = filedialog.askopenfilename(title="Select One Signal File", filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return

        time_data, amplitude_data = load_signal_from_file_task2(file_path)
        if time_data is None or amplitude_data is None:
            return

        if operation == "Multiplication":
            constant = float(multiplication_entry.get())
            result_amplitudes = multiply_signal(amplitude_data, constant)
        elif operation == "Square":
            result_amplitudes = square_amplitudes(amplitude_data)
        elif operation == "Normalize":
            to_zero_one = normalize_var.get() == 1
            result_amplitudes = normalize_signal(amplitude_data, to_zero_one=to_zero_one)
        elif operation == "Accumulation":
            result_amplitudes = accumulation_amplitudes(amplitude_data)

    ax1_task2.clear()
    ax2_task2.clear()

    ax1_task2.plot(time_data, amplitude_data, color='cyan', marker='o', markersize=5)
    ax1_task2.set_facecolor('black')
    ax1_task2.set_title('Original Signal', color='white')
    ax1_task2.set_xlabel('Time (s)', color='white')
    ax1_task2.set_ylabel('Amplitude', color='white')
    ax1_task2.tick_params(colors='red')

    ax2_task2.plot(time_data, result_amplitudes, color='magenta', marker='x', markersize=5)
    ax2_task2.set_facecolor('black')
    ax2_task2.set_title(f'{operation} Result', color='white')
    ax2_task2.set_xlabel('Time (s)', color='white')
    ax2_task2.set_ylabel('Amplitude', color='white')
    ax2_task2.tick_params(colors='red')

    canvas_task2.draw()

    expected_file_path = filedialog.askopenfilename(title="Select Expected Output File",
                                                    filetypes=[("Text Files", "*.txt")])
    if expected_file_path:
        SignalSamplesAreEqual(expected_file_path, time_data, result_amplitudes)


# --- Task 3 Functions ---
def load_signal_from_file_task3():
    file_path = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            num_samples = int(lines[2].strip())
            amplitude_data = [float(line.split()[1]) for line in lines[3:3 + num_samples]]
        return amplitude_data
    return None


def quantize_signal(amplitude_data, num_bits=None, num_levels=None):
    if num_bits is not None:
        num_levels = 2 ** num_bits

    min_val, max_val = min(amplitude_data), max(amplitude_data)
    delta = np.round((max_val - min_val) / num_levels, decimals=3)

    midpoints = [min_val + delta * (i + 0.5) for i in range(num_levels)]
    interval_indices = []
    encoded_values = []
    quantized_values = []
    sampled_errors = []

    for sample in amplitude_data:
        closest_index = int((sample - min_val) // delta)
        closest_index = min(closest_index, num_levels - 1)
        interval_index = closest_index + 1
        quantized_value = midpoints[closest_index]
        encoded_value = format(closest_index, f'0{len(bin(num_levels - 1)) - 2}b')

        interval_indices.append(interval_index)
        encoded_values.append(encoded_value)
        quantized_values.append(quantized_value)
        sampled_errors.append(quantized_value - sample)

    return interval_indices, encoded_values, quantized_values, sampled_errors


def display_task3_results(interval_indices, encoded_values, quantized_values, sampled_errors, show_error=False):
    data = {
        "Interval Index": interval_indices,
        "Encoded Values": encoded_values,
        "Quantized Signal": quantized_values,
    }
    if show_error:
        data["Quantization Error"] = sampled_errors

    df = pd.DataFrame(data)

    for widget in table_frame.winfo_children():
        widget.destroy()

    for col_index, col_name in enumerate(df.columns):
        label = tk.Label(table_frame, text=col_name, font=("Helvetica", 10, "bold"))
        label.grid(row=0, column=col_index)

    for row_index, row in df.iterrows():
        for col_index, value in enumerate(row):
            label = tk.Label(table_frame, text=f"{value:.4f}" if isinstance(value, float) else value)
            label.grid(row=row_index + 1, column=col_index)


def QuantizationTest1(file_name, Your_EncodedValues, Your_QuantizedValues):
    expectedEncodedValues = []
    expectedQuantizedValues = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V2 = str(L[0])
                V3 = float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if ((len(Your_EncodedValues) != len(expectedEncodedValues)) or (
            len(Your_QuantizedValues) != len(expectedQuantizedValues))):
        print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if (Your_EncodedValues[i] != expectedEncodedValues[i]):
            print(
                "QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print(
                "QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one")
            return
    print("QuantizationTest1 Test case passed successfully")


def QuantizationTest2(file_name, Your_IntervalIndices, Your_EncodedValues, Your_QuantizedValues, Your_SampledError):
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 4:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = str(L[1])
                V3 = float(L[2])
                V4 = float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if (len(Your_IntervalIndices) != len(expectedIntervalIndices)
            or len(Your_EncodedValues) != len(expectedEncodedValues)
            or len(Your_QuantizedValues) != len(expectedQuantizedValues)
            or len(Your_SampledError) != len(expectedSampledError)):
        print("QuantizationTest2 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_IntervalIndices)):
        if (Your_IntervalIndices[i] != expectedIntervalIndices[i]):
            print("QuantizationTest2 Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(Your_EncodedValues)):
        if (Your_EncodedValues[i] != expectedEncodedValues[i]):
            print(
                "QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return

    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print(
                "QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one")
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one")
            return
    print("QuantizationTest2 Test case passed successfully")


def on_calculate_task3():
    amplitude_data = load_signal_from_file_task3()
    if amplitude_data is None:
        return

    try:

        num_bits = bits_entry.get()
        num_levels = levels_entry.get()

        if num_bits:

            num_bits = int(num_bits)
            interval_indices, encoded_values, quantized_values, _ = quantize_signal(amplitude_data, num_bits=num_bits)
            display_task3_results(interval_indices, encoded_values, quantized_values, [], show_error=False)

            expected_file_path = filedialog.askopenfilename(title="Select Expected Output File",
                                                            filetypes=[("Text Files", "*.txt")])
            if expected_file_path:
                QuantizationTest1(expected_file_path, encoded_values, quantized_values)

        elif num_levels:

            num_levels = int(num_levels)
            interval_indices, encoded_values, quantized_values, sampled_errors = quantize_signal(amplitude_data,
                                                                                                 num_levels=num_levels)
            display_task3_results(interval_indices, encoded_values, quantized_values, sampled_errors, show_error=True)

            expected_file_path = filedialog.askopenfilename(title="Select Expected Output File",
                                                            filetypes=[("Text Files", "*.txt")])
            if expected_file_path:
                QuantizationTest2(expected_file_path, interval_indices, encoded_values, quantized_values,
                                  sampled_errors)

    except ValueError as e:
        print(f"Error: {str(e)}")


# --- Task 4 Functions ---
def load_signal_from_file_task4():
    file_path = filedialog.askopenfilename(title="Select Input File", filetypes=[("Text Files", "*.txt")])
    if not file_path:
        return None, None, None

    with open(file_path, 'r') as file:
        lines = file.readlines()
        signal_type = int(lines[0].strip())  # 0 for DFT, 1 for IDFT
        is_periodic = int(lines[1].strip())  # 0 for periodic, 1 for non-periodic
        N = int(lines[2].strip())  # Number of samples
        signal_data = []

        # Read signal data based on the signal type
        for line in lines[3:3 + N]:
            if signal_type == 0:  # DFT input
                # Split by space for DFT
                values = line.strip().split()
                if len(values) >= 2:
                    sample_value = values[1]  # Ignore the index (values[0])
                    # Convert to float and append
                    amplitude = float(sample_value.strip())
                    signal_data.append(amplitude)
            elif signal_type == 1:  # IDFT input
                # Split by comma for IDFT and handle 'f' suffix
                values = line.strip().split(',')
                if len(values) >= 2:
                    amplitude = float(values[0].replace('f', '').strip())
                    phase_shift = float(values[1].replace('f', '').strip())
                    # Convert to complex form and append
                    signal_data.append(complex(amplitude * np.cos(phase_shift), amplitude * np.sin(phase_shift)))

    return signal_type, is_periodic, signal_data


def load_expected_output(file_path):
    expected_amplitude = []
    expected_phase_shift = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        signal_type = int(lines[0].strip())  # 0 for IDFT, 1 for DFT
        num_samples = int(lines[2].strip())  # Number of samples

        for line in lines[3:3 + num_samples]:
            if signal_type == 1:  # DFT output format with amplitude and phase shift
                # Split and remove 'f' suffix if present
                parts = [value.replace('f', '').strip() for value in line.split()]
                if len(parts) >= 2:
                    amplitude = float(parts[0])  # Amplitude
                    phase_shift = float(parts[1])  # Phase shift
                    expected_amplitude.append(amplitude)
                    expected_phase_shift.append(phase_shift)
            elif signal_type == 0:  # IDFT output format with only amplitude
                # Split by space, remove 'f' suffix from amplitude if present
                parts = line.split()
                if len(parts) >= 2:
                    amplitude = float(parts[1].replace('f', '').strip())  # Ignore the index (parts[0])
                    expected_amplitude.append(amplitude)
                    expected_phase_shift.append(0)  # Default phase shift to 0 for IDFT

    return expected_amplitude, expected_phase_shift


# Function to compute DFT manually
def compute_transform(signal_data, N, inverse=False):
    result = []
    for k in range(N):
        if inverse:
            # IDFT calculation
            real_part = sum(signal_data[n] * np.cos(2 * np.pi * k * n / N) for n in range(N)) / N
            imag_part = sum(signal_data[n] * np.sin(2 * np.pi * k * n / N) for n in range(N)) / N
        else:
            # DFT calculation
            real_part = sum(signal_data[n] * np.cos(2 * np.pi * k * n / N) for n in range(N))
            imag_part = -sum(signal_data[n] * np.sin(2 * np.pi * k * n / N) for n in range(N))
        result.append(complex(real_part, imag_part))
    return result


# Use to test the Amplitude of DFT and IDFT
def SignalComapreAmplitude(SignalInput=[], SignalOutput=[]):  # origin #elle ana 3amalto

    # print (SignalInput , SignalOutput )
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            if abs(SignalInput[i] - SignalOutput[i]) > 0.001:
                return False
            elif SignalInput[i] != SignalOutput[i]:
                return False
        return True


def RoundPhaseShift(P):
    while P < 0:
        P += 2 * math.pi
    return float(P % (2 * math.pi))


# Use to test the PhaseShift of DFT
def SignalComaprePhaseShift(SignalInput=[], SignalOutput=[]):
    # print (SignalInput,SignalOutput)
    if len(SignalInput) != len(SignalInput):
        return False
    else:
        for i in range(len(SignalInput)):
            A = round(SignalInput[i])
            B = round(SignalOutput[i])
            if abs(A - B) > 0.0001:
                return False
            elif A != B:
                return False
        return True


def on_calculate_task4():
    # Clear previous content on existing plot axes
    ax1.clear()
    ax2.clear()

    # Load signal file
    signal_type, is_periodic, signal_data = load_signal_from_file_task4()
    if signal_data is None:
        return

    # Get sampling frequency from the entry field
    try:
        sampling_freq = float(sampling_freq_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid sampling frequency.")
        return

    N = len(signal_data)

    # Determine if DFT or IDFT based on the first line of the file
    inverse = signal_type == 1
    transform_result = compute_transform(signal_data, N, inverse)

    if inverse:
        # For IDFT, we do not plot, just compare amplitude
        # expected_file_path = filedialog.askopenfilename(title="Select Expected Output File",
        #                                                 filetypes=[("Text Files", "*.txt")])
        # if not expected_file_path:
        #     return
        #
        # expected_amplitude, _ = load_expected_output(expected_file_path)  # Ignore phase shift data for IDFT

        # Calculate absolute values and round both expected and calculated amplitudes
        rounded_calculated_amplitude = [round(abs(comp), 4) for comp in transform_result]
        # rounded_expected_amplitude = [round(a, 4) for a in expected_amplitude]
        print(rounded_calculated_amplitude)

        # Perform amplitude comparison only
        # amplitude_test = SignalComapreAmplitude(rounded_expected_amplitude, rounded_calculated_amplitude)
        # amplitude_message = "Amplitude Test Passed" if amplitude_test else "Amplitude Test Failed"

        # Display amplitude comparison result only
        # messagebox.showinfo("Comparison Results", amplitude_message)

    else:
        # For DFT, calculate amplitude and phase shift
        amplitude = [np.sqrt(comp.real * 2 + comp.imag * 2) for comp in transform_result]
        phase_shift = [np.arctan2(comp.imag, comp.real) for comp in transform_result]
        Ts = 1 / sampling_freq  # Time interval
        omega = [2 * np.pi * k / (N * Ts) for k in range(N)]

        # Discrete (Stem) Amplitude Plot
        ax1.stem(omega, amplitude, basefmt=" ", linefmt="cyan", markerfmt="bo")
        ax1.set_facecolor("black")
        ax1.set_title("Amplitude vs. Omega - Discrete (Stem)", color="white")
        ax1.set_xlabel("Omega (rad/s)", color="white")
        ax1.set_ylabel("Amplitude", color="white")
        ax1.tick_params(colors="red")

        # Discrete (Stem) Phase Shift Plot
        ax2.stem(omega, phase_shift, basefmt=" ", linefmt="magenta", markerfmt="mo")
        ax2.set_facecolor("black")
        ax2.set_title("Phase Shift vs. Omega - Discrete (Stem)", color="white")
        ax2.set_xlabel("Omega (rad/s)", color="white")
        ax2.set_ylabel("Phase Shift (radians)", color="white")
        ax2.tick_params(colors="red")

        # Redraw the canvas with updated plots
        canvas_task4.draw()
        print(amplitude)
        print(phase_shift)

        # Load expected values from the output file after plotting
        # expected_file_path = filedialog.askopenfilename(title="Select Expected Output File",
        #                                                 filetypes=[("Text Files", "*.txt")])
        # if not expected_file_path:
        #     return
        #
        # expected_amplitude, expected_phase_shift = load_expected_output(expected_file_path)

        # Round values before comparison
        # rounded_calculated_amplitude = [round(a, 4) for a in amplitude]
        # rounded_expected_amplitude = [round(a, 4) for a in expected_amplitude]

        # Perform comparisons
        # amplitude_test = SignalComapreAmplitude(rounded_expected_amplitude, rounded_calculated_amplitude)
        # rounded_calculated_phase_shift = [round(p, 4) for p in phase_shift]
        # rounded_expected_phase_shift = [round(p, 4) for p in expected_phase_shift]

        # phase_test = SignalComaprePhaseShift(rounded_expected_phase_shift, rounded_calculated_phase_shift)

        # Display comparison results
        # amplitude_message = "Amplitude Test Passed" if amplitude_test else "Amplitude Test Failed"
        # phase_message = "Phase Shift Test Passed" if phase_test else "Phase Shift Test Failed"
        # messagebox.showinfo("Comparison Results", f"{amplitude_message}\n{phase_message}")


# --- Test Functions ---
def SignalSamplesAreEqual(file_name, indices, samples):
    """
    Compare the processed signal's indices and samples with the expected output file.
    """
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        # Skip the first three lines
        f.readline()  # Signal Type
        f.readline()  # Periodicity
        f.readline()  # Number of Samples

        # Read expected indices and samples
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                expected_indices.append(int(parts[0]))
                expected_samples.append(float(parts[1]))

    # Check if lengths match
    if len(expected_samples) != len(samples):
        print("Test case failed: Your signal has a different length from the expected one.")
        return False

    # Check values
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) > 0.01:
            print(f"Test case failed: Sample mismatch at index {i}. Expected {expected_samples[i]}, Got {samples[i]}.")
            return False

    print("Test case passed successfully.")
    return True


# --- Task 5 Functions ---

# Function to compute the DCT
def compute_dct(input_signal):
    N = len(input_signal)
    dct_result = []
    normalization_factor = np.sqrt(2 / N)

    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += input_signal[n] * np.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
        y_k = normalization_factor * sum_val
        dct_result.append(y_k)

    return np.array(dct_result)


# Function to compare the output signal with expected output
def SignalSamplesAreEqual(file_name, indices, samples):
    """
    Compare the processed signal's indices and samples with the expected output file.
    """
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        # Skip the first three lines
        f.readline()  # Signal Type
        f.readline()  # Periodicity
        f.readline()  # Number of Samples

        # Read expected indices and samples
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                expected_indices.append(int(parts[0]))
                expected_samples.append(float(parts[1]))

    # Check if lengths match
    if len(expected_samples) != len(samples):
        print("Test case failed: Your signal has a different length from the expected one.")
        return False

    # Check values
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) > 0.01:
            print(f"Test case failed: Sample mismatch at index {i}. Expected {expected_samples[i]}, Got {samples[i]}.")
            return False

    print("Test case passed successfully.")
    return True


def on_calculate_task5_dct():
    # Step 1: Load the input file
    input_file_path = filedialog.askopenfilename(
        title="Select Input File",
        filetypes=[("Text Files", "*.txt")]
    )
    if not input_file_path:
        print("No input file selected.")
        return

    try:
        # Step 2: Read the input signal from the file
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
            num_samples = int(lines[2].strip())  # Number of samples
            input_signal = [float(line.split()[1]) for line in lines[3:3 + num_samples]]  # Extract signal values

        # Step 3: Get the number of coefficients from the user
        try:
            num_coefficients = int(coefficients_entry.get())
            if num_coefficients <= 0 or num_coefficients > num_samples:
                messagebox.showerror("Invalid Input",
                                     "Number of coefficients must be between 1 and the number of samples.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for the number of coefficients.")
            return

        # Step 4: Compute the DCT
        dct_result = compute_dct(input_signal)[:num_coefficients]

        # Step 5: Plot the DCT result
        ax_task5.clear()
        ax_task5.plot(range(num_coefficients), dct_result, 'r-o', label="DCT Coefficients")
        ax_task5.set_facecolor("white")
        ax_task5.set_title("DCT Coefficients", color="black")
        ax_task5.set_xlabel("Coefficient Index", color="black")
        ax_task5.set_ylabel("Coefficient Value", color="black")
        ax_task5.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        ax_task5.legend()
        canvas_task5.draw()

        # Step 6: Select the expected output file
        expected_file_path = filedialog.askopenfilename(
            title="Select Expected Output File",
            filetypes=[("Text Files", "*.txt")]
        )
        if not expected_file_path:
            print("No expected output file selected.")
            return

        # Step 7: Compare with the expected output
        SignalSamplesAreEqual(expected_file_path, list(range(num_coefficients)), dct_result)

        # Step 8: Save the output in a new text file
        output_file_path = filedialog.asksaveasfilename(
            title="Save Output File",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if output_file_path:
            with open(output_file_path, 'w') as output_file:
                output_file.write(f"0\n0\n{num_coefficients}\n")
                for i, value in enumerate(dct_result):
                    output_file.write(f"{i} {value:.6f}\n")
            print(f"DCT output saved to {output_file_path}")

    except FileNotFoundError:
        messagebox.showerror("Error", "File not found.")
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid file format or data: {e}")


def DerivativeSignal():
    # Input signal
    InputSignal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                   53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                   78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

    # Expected output for validation
    expectedOutput_first = [1] * (len(InputSignal) - 1)
    expectedOutput_second = [0] * (len(InputSignal) - 2)

    FirstDrev = [InputSignal[n] - InputSignal[n - 1] for n in range(1, len(InputSignal))]
    SecondDrev = [InputSignal[n + 1] - 2 * InputSignal[n] + InputSignal[n - 1] for n in range(1, len(InputSignal) - 1)]

    # Check length mismatch
    if (len(FirstDrev) != len(expectedOutput_first)) or (len(SecondDrev) != len(expectedOutput_second)):
        print("Mismatch in length")
        return

    # Check for derivative correctness
    first = second = True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
            continue
        else:
            first = False
            print("1st derivative wrong")
            return

    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
            continue
        else:
            second = False
            print("2nd derivative wrong")
            return

    if first and second:
        print("Derivative Test case passed successfully")
    else:
        print("Derivative Test case failed")

    # Plot the original signal
    ax_task5.plot(range(len(InputSignal)), InputSignal, 'b-', label="Original Signal", linewidth=2)

    # Plot the first derivative
    ax_task5.plot(range(1, len(FirstDrev) + 1), FirstDrev, 'r-o', label="1st Derivative")

    # Plot the second derivative
    ax_task5.plot(range(2, len(SecondDrev) + 2), SecondDrev, 'g-s', label="2nd Derivative")

    ax_task5.set_facecolor("white")
    ax_task5.set_title("Original Signal and Derivatives", color="black")
    ax_task5.set_xlabel("Index", color="black")
    ax_task5.set_ylabel("Value", color="black")
    ax_task5.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    ax_task5.legend()

    # Redraw the canvas
    canvas_task5.draw()


def fold_signal(signal):
    """Reverse the signal values."""
    folded_signal = []
    for i in range(len(signal) - 1, -1, -1):  # Loop from the end to the start
        folded_signal.append(signal[i])
    return folded_signal


def Shift_Fold_Signal(file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Shift_Fold_Signal Test case failed, your signal have different values from the expected one")
            return
    print("Shift_Fold_Signal Test case passed successfully")


def on_calculate_task5_folding():
    """Handle folding and validation tasks."""

    # Step 2: Load the signal
    signal_type, is_periodic, time_data, signal = load_signal_from_file()
    if signal is None or time_data is None:
        print("Failed to load signal data.")
        return

    # Step 3: Fold the signal
    folded_signal = fold_signal(signal)

    # Step 4: Plot both the original and folded signals
    ax_task5.clear()

    # Plot the original signal
    ax_task5.plot(time_data, signal, label="Original Signal", color="blue", linestyle='-', marker='o')

    # Plot the folded signal
    ax_task5.plot(time_data, folded_signal, label="Folded Signal", color="red", linestyle='--', marker='x')

    ax_task5.set_title("Original and Folded Signals", color="black")
    ax_task5.set_xlabel("Time", color="black")
    ax_task5.set_ylabel("Amplitude", color="black")
    ax_task5.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    ax_task5.legend()

    canvas_task5.draw()

    # Step 5: Load the expected output file
    expected_file_path = filedialog.askopenfilename(
        title="Select Expected Output File",
        filetypes=[("Text Files", "*.txt")]
    )
    if not expected_file_path:
        print("No expected output file selected.")
        return

    # Step 6: Compare with the expected output
    SignalSamplesAreEqual(expected_file_path, time_data, folded_signal)
def adjust_time_data(time_data):
    return [t + 500 for t in time_data]


def on_calculate_task5_pshiftfolding():
    """Handle folding and validation tasks."""

    signal_type, is_periodic, time_data, signal = load_signal_from_file()
    if signal is None or time_data is None:
        print("Failed to load signal data.")
        return

    time_positive = adjust_time_data(time_data)

    ax_task5.clear()

    ax_task5.plot(time_data, signal, label="Original Signal", color="blue", linestyle='-', marker='o')

    ax_task5.plot(time_positive, signal, label="Folded Signal", color="red", linestyle='--', marker='x')
    ax_task5.set_title("Original and Folded Signals", color="black")
    ax_task5.set_xlabel("Time", color="black")
    ax_task5.set_ylabel("Amplitude", color="black")
    ax_task5.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Add legend
    ax_task5.legend()

    # Redraw the canvas
    canvas_task5.draw()
    # Step 4: Load the expected output file
    expected_file_path = filedialog.askopenfilename(
        title="Select Expected Output File",
        filetypes=[("Text Files", "*.txt")]
    )
    if not expected_file_path:
        print("No expected output file selected.")
        return
    Shift_Fold_Signal(expected_file_path, time_positive, signal)


def adjust_time_data_negative(time_data):
    return [t - 500 for t in time_data]


def on_calculate_task5_shiftfolding():
    """Handle folding and validation tasks."""

    signal_type, is_periodic, time_data, signal = load_signal_from_file()
    if signal is None or time_data is None:
        print("Failed to load signal data.")
        return

    # Step 3: Fold the signal
    time_negative = adjust_time_data_negative(time_data)

    ax_task5.clear()

    ax_task5.plot(time_data, signal, label="Original Signal", color="blue", linestyle='-', marker='o')

    ax_task5.plot(time_negative, signal, label="Folded Signal", color="red", linestyle='--', marker='x')
    ax_task5.set_title("Original and Folded Signals", color="black")
    ax_task5.set_xlabel("Time", color="black")
    ax_task5.set_ylabel("Amplitude", color="black")
    ax_task5.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Step 4: Load the expected output file
    expected_file_path = filedialog.askopenfilename(
        title="Select Expected Output File",
        filetypes=[("Text Files", "*.txt")]
    )
    if not expected_file_path:
        print("No expected output file selected.")
        return
    Shift_Fold_Signal(expected_file_path, time_negative, signal)


# Function to load signal data from a file
def load_signal_from_file(file_path):
    """Load signal data from the given file, skipping the first three metadata lines."""
    time_data = []
    signal = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

        # Skip the first three metadata lines
        for line in lines[3:]:  # Start reading from the 4th line
            if line.strip():  # Ensure the line is not empty
                index, amplitude = line.split()  # Split the line into index and amplitude
                time_data.append(int(index))  # Assuming index is an integer
                signal.append(float(amplitude))  # Assuming amplitude is a float

    return time_data, signal


def choose_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])
    return file_path


#
def load_signal_from_file():
    file_path = choose_file()
    if not file_path:
        print("No file selected.")
        return None, None, None

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # Read the metadata from the file
            signal_type = lines[0].strip().split()[0]
            is_periodic = int(lines[1].strip())
            num_samples = int(lines[2].strip())

            # Initialize lists to store time data and signal data
            time_data = []
            signal = []

            # Read the time and signal values
            for line in lines[3:3 + num_samples]:
                parts = list(map(float, line.strip().split()))
                time_data.append(parts[0])  # First column is time
                signal.append(parts[1])  # Second column is signal value

        return signal_type, is_periodic, time_data, signal

    except FileNotFoundError:
        print("File not found.")
    except ValueError:
        print("Error reading data from file.")

    return None, None, None, None


def delay_signal(k_entry):
    """
    Delays the signal by k steps and updates the plot in the Task 5 frame.
    """
    try:
        k = int(k_entry.get())  # Get the delay steps
    except ValueError:
        messagebox.showerror("Error", "Invalid value for k! Please enter an integer.")
        return

    # Define the original signal (example)
    indices = np.arange(0, 10)  # Example indices
    amplitudes = np.sin(indices)  # Example signal: sine wave values

    # Delay the signal
    delayed_indices = indices + k

    # Plot the original and delayed signals
    ax_task5.clear()
    ax_task5.plot(indices, amplitudes, label="Original Signal", marker="o", color="blue")
    ax_task5.plot(delayed_indices, amplitudes, label=f"Delayed Signal by {k} Steps", marker="x", color="red")
    ax_task5.set_title(f"Signal Delayed by {k} Steps", color="white")
    ax_task5.set_xlabel("Indices", color="white")
    ax_task5.set_ylabel("Amplitude", color="white")
    ax_task5.legend()
    ax_task5.grid(True)
    ax_task5.set_facecolor('white')
    ax_task5.tick_params(colors="black")

    canvas_task5.draw()


def advance_signal(k_entry):
    """
    Advances the signal by k steps and updates the plot in the Task 5 frame.
    """
    try:
        k = int(k_entry.get())  # Get the advance steps
    except ValueError:
        messagebox.showerror("Error", "Invalid value for k! Please enter an integer.")
        return

    # Define the original signal (example)
    indices = np.arange(0, 10)  # Example indices
    amplitudes = np.sin(indices)  # Example signal: sine wave values

    # Advance the signal
    advanced_indices = indices - k

    # Plot the original and advanced signals
    ax_task5.clear()
    ax_task5.plot(indices, amplitudes, label="Original Signal", marker="o", color="blue")
    ax_task5.plot(advanced_indices, amplitudes, label=f"Advanced Signal by {k} Steps", marker="x", color="green")
    ax_task5.set_title(f"Signal Advanced by {k} Steps", color="white")
    ax_task5.set_xlabel("Indices", color="white")
    ax_task5.set_ylabel("Amplitude", color="white")
    ax_task5.legend()
    ax_task5.grid(True)
    ax_task5.set_facecolor('white')
    ax_task5.tick_params(colors="black")

    canvas_task5.draw()


# --- Task 6 Functions ---

def smoothing_signal(window_size):
    """Compute the moving average of a signal with a specified window size."""
    try:
        window_size = int(window_size.strip())
    except ValueError:
        tk.messagebox.showerror("Error", "Invalid window size! Please enter a positive integer.")
        return None

    if window_size <= 0:
        tk.messagebox.showerror("Error", "Window size must be greater than 0.")
        return None

    # Step 1: Load the input signal
    signal_type, is_periodic, indices, samples = load_signal_from_file()
    if indices is None or samples is None:
        print("Failed to load signal data.")
        return None

    # Step 2: Calculate the moving average
    smoothed_samples = []
    for i in range(len(samples) - window_size + 1):
        window = samples[i:i + window_size]
        smoothed_samples.append(sum(window) / window_size)

    # Adjust the indices to match the output signal
    smoothed_indices = indices[:len(smoothed_samples)]

    # Step 3: Plot the results
    plot_smoothing(indices, samples, smoothed_indices, smoothed_samples)

    # Step 4: Compare with the expected output file
    expected_file_path = filedialog.askopenfilename(
        title="Select Expected Output File",
        filetypes=[("Text Files", "*.txt")]
    )
    if not expected_file_path:
        print("No expected output file selected.")
        return

    SignalSamplesAreEqual(expected_file_path, smoothed_indices, smoothed_samples)


def plot_smoothing(original_indices, original_samples, smoothed_indices, smoothed_samples):
    """Plot the original and smoothed signals."""

    # Clear the previous plot
    ax_task6.clear()  # Clear the axis (not the canvas)

    # Plot the original and smoothed signals
    ax_task6.plot(original_indices, original_samples, label="Original Signal", marker="o", linestyle="-", color="blue")
    ax_task6.plot(smoothed_indices, smoothed_samples, label="Smoothed Signal", marker="x", linestyle="--", color="red")

    # Customize the plot
    ax_task6.set_title("Smoothing (Moving Average)")
    ax_task6.set_xlabel("Indices")
    ax_task6.set_ylabel("Amplitude")
    ax_task6.legend(loc="upper right")
    ax_task6.grid(True)

    # Redraw the canvas to update the plot
    canvas_task6.draw()


def discrete_convolution(time_data1, signal1, time_data2, signal2):
    """Perform discrete convolution of two signals."""
    if signal1 is None or signal2 is None or time_data1 is None or time_data2 is None:
        print("Failed to load signal data.")
        return [], []

    # Calculate the range of output indices
    n_min = int(min(time_data1) + min(time_data2))  # Ensure n_min is an integer
    n_max = int(max(time_data1) + max(time_data2))  # Ensure n_max is an integer
    output_indices = list(range(n_min, n_max + 1))

    # Create dictionaries for signal lookup by index
    x_signal = dict(zip(time_data1, signal1))
    h_signal = dict(zip(time_data2, signal2))

    # Perform convolution
    output_samples = []
    for n in output_indices:
        convolution_sum = 0
        for k in time_data1:
            h_index = n - k
            # Check if h_index exists in h_signal
            if h_index in h_signal:
                convolution_sum += x_signal[k] * h_signal[h_index]
        output_samples.append(convolution_sum)

    return output_indices, output_samples


def ConvTest(Your_indices, Your_samples):
    """
    Test inputs
    InputIndicesSignal1 =[-2, -1, 0, 1]
    InputSamplesSignal1 = [1, 2, 1, 1 ]

    InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
    InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
    """

    expected_indices = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1]

    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Conv Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Conv Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Conv Test case failed, your signal have different values from the expected one")
            return
    print("Conv Test case passed successfully")


# def convolution_task():
#     time_data1, signal1=load_signal_from_file1()
#     time_data2, signal2=load_signal_from_file2()
#     Your_indices, Your_samples=discrete_convolution(time_data1, signal1, time_data2, signal2)
#     ConvTest(Your_indices, Your_samples)
def convolution_task():
    signal_type1, is_periodic1, time_data1, signal1 = load_signal_from_file()

    signal_type2, is_periodic2, time_data2, signal2 = load_signal_from_file()

    output_indices, output_samples = discrete_convolution(time_data1, signal1, time_data2, signal2)
    ConvTest(output_indices, output_samples)


def choose_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])
    return file_path


#
def load_signal_from_file():
    file_path = choose_file()
    if not file_path:
        print("No file selected.")
        return None, None, None

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # Read the metadata from the file
            signal_type = lines[0].strip().split()[0]
            is_periodic = int(lines[1].strip())
            num_samples = int(lines[2].strip())

            # Initialize lists to store time data and signal data
            time_data = []
            signal = []

            # Read the time and signal values
            for line in lines[3:3 + num_samples]:
                parts = list(map(float, line.strip().split()))
                time_data.append(parts[0])  # First column is time
                signal.append(parts[1])  # Second column is signal value

        return signal_type, is_periodic, time_data, signal

    except FileNotFoundError:
        print("File not found.")
    except ValueError:
        print("Error reading data from file.")

    return None, None, None, None


def normalized_cross_correlation(X1, X2):
    X1 = np.array(X1)
    X2 = np.array(X2)

    N = len(X1)  # Signal length

    # Pre-compute squared sums for normalization
    X1_squared_sum = np.sum(X1 ** 2)
    X2_squared_sum = np.sum(X2 ** 2)
    normalization = np.sqrt(X1_squared_sum * X2_squared_sum)

    # Compute the cross-correlation numerator
    r12 = []
    for j in range(N):
        numerator = sum(X1[i] * X2[(i + j) % N] for i in range(N))  # Periodic signals
        r12.append(numerator / normalization)

    return np.array(r12)


# Compute normalized cross-correlation

def choose_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])
    return file_path


#
def load_signal_from_file():
    file_path = choose_file()
    if not file_path:
        print("No file selected.")
        return None, None, None

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # Read the metadata from the file
            signal_type = lines[0].strip().split()[0]
            is_periodic = int(lines[1].strip())
            num_samples = int(lines[2].strip())

            # Initialize lists to store time data and signal data
            time_data = []
            signal = []

            # Read the time and signal values
            for line in lines[3:3 + num_samples]:
                parts = list(map(float, line.strip().split()))
                time_data.append(parts[0])  # First column is time
                signal.append(parts[1])  # Second column is signal value

        return signal_type, is_periodic, time_data, signal

    except FileNotFoundError:
        print("File not found.")
    except ValueError:
        print("Error reading data from file.")

    return None, None, None, None


def Compare_Signals(file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Correlation Test case failed, your signal have different values from the expected one")
            return
    print("Correlation Test case passed successfully")


def calculate_correlation():
    signal_type, is_periodic, time_data, signal1 = load_signal_from_file()
    signal_type, is_periodic, time_data, signal2 = load_signal_from_file()

    r12 = normalized_cross_correlation(signal1, signal2)

    expected_file_path = filedialog.askopenfilename(
        title="Select Expected Output File",
        filetypes=[("Text Files", "*.txt")]
    )
    if not expected_file_path:
        print("No expected output file selected.")
        return

    Compare_Signals(expected_file_path, time_data, r12)


def remove_dc_time_domain(signal):
    # Calculate the mean value (DC component)
    mean_value = sum(signal) / len(signal)

    # Remove the DC component and round each value to 3 decimal places
    signal_without_dc = [round(sample - mean_value, 3) for sample in signal]

    return signal_without_dc


def calc_dc():
    signal_type, is_periodic, time_data, input_signal = load_signal_from_file()
    signal_without_dc = remove_dc_time_domain(input_signal)
    expected_file_path = filedialog.askopenfilename(
        title="Select Expected Output File",
        filetypes=[("Text Files", "*.txt")]
    )
    if not expected_file_path:
        print("No expected output file selected.")
        return

    # Step 6: Compare with the expected output
    SignalSamplesAreEqual(expected_file_path, time_data, signal_without_dc)


def remove_dc_frequency_dft_idft():
    # Step 1: Load the input signal
    signal_type, is_periodic, indices, samples = load_signal_from_file()
    if indices is None or samples is None:
        print("Failed to load signal data.")
        return

    # Step 2: Perform DFT
    N = len(samples)
    dft_result = np.fft.fft(samples)

    # Step 3: Remove the DC component (set the first harmonic to 0)
    dft_result[0] = 0

    # Step 4: Perform IDFT
    idft_result = np.fft.ifft(dft_result).real

    # Step 5: Plot the results
    plot_remove_dc(indices, samples, idft_result)

    # Step 6: Compare with the expected output file
    expected_file_path = filedialog.askopenfilename(
        title="Select Expected Output File",
        filetypes=[("Text Files", "*.txt")]
    )
    if not expected_file_path:
        print("No expected output file selected.")
        return

    SignalSamplesAreEqual(expected_file_path, indices, idft_result)


def plot_remove_dc(original_indices, original_signal, signal_without_dc):
    # Clear the previous plot
    ax_task6.clear()  # Clear the axis (not the canvas)

    # Plot the original and DC-removed signals
    ax_task6.plot(original_indices, original_signal, label="Original Signal", marker="o", linestyle="-", color="blue")
    ax_task6.plot(original_indices, signal_without_dc, label="Signal without DC", marker="x", linestyle="--",
                  color="red")

    # Customize the plot
    ax_task6.set_title("Removing DC Component Using DFT and IDFT")
    ax_task6.set_xlabel("Indices")
    ax_task6.set_ylabel("Amplitude")
    ax_task6.legend(loc="upper right")
    ax_task6.grid(True)

    # Redraw the canvas to update the plot
    canvas_task6.draw()


# --- Task 7 Functions ---

def read_ecg_file(filename):
    indices = []
    samples = []
    try:
        with open(filename, 'r') as f:
            # Skip the first 3 lines (metadata)
            for _ in range(3):
                f.readline()
            # Read the rest of the lines
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:  # Ensure it's a valid index-sample pair
                    indices.append(int(parts[0]))
                    samples.append(float(parts[1]))
    except Exception as e:
        print(f"Error reading ECG file: {e}")
    return indices, samples


def filter_signal_task7():
    filter_type = filter_type_menu.get()  # Low Pass, High Pass, Band Pass, Band Stop
    sampling_freq = float(sampling_freq_entry.get())  # Sampling frequency in Hz
    cutoff_freq = float(cutoff_freq_entry.get())  # Cutoff frequency in Hz
    f2 = float(second_cutoff_freq_entry.get())
    stop_atten = float(stop_atten_entry.get())  # Stop band attenuation (dB)
    transition_band = float(transition_band_entry.get())  # Transition band in Hz

    delta_f = transition_band / sampling_freq

    if 1 <= stop_atten <= 21:
        window_type = "Rectangular"
        window_factor = 0.9
    elif 22 <= stop_atten <= 44:
        window_type = "Hanning"
        window_factor = 3.1
    elif 45 <= stop_atten <= 53:
        window_type = "Hamming"
        window_factor = 3.3
    elif 54 <= stop_atten <= 74:
        window_type = "Blackman"
        window_factor = 5.5
    else:
        raise ValueError("Stop band attenuation out of range for known windows.")
    N = window_factor / delta_f
    N = math.ceil(N)
    if N % 2 == 0:
        N += 1
    n = np.arange(-(N // 2), N // 2 + 1)
    hd = np.zeros_like(n, dtype=float)

    if filter_type == "Low Pass":
        fc_normalized = (cutoff_freq + (transition_band / 2)) / sampling_freq
        wc = 2 * np.pi * fc_normalized
        hd[n == 0] = 2 * fc_normalized
        hd[n != 0] = (2 * fc_normalized) * (np.sin(n[n != 0] * wc) / (n[n != 0] * wc))

    elif filter_type == "High Pass":
        fc_normalized = (cutoff_freq - (transition_band / 2)) / sampling_freq
        wc = 2 * np.pi * fc_normalized
        hd[n == 0] = (1 - (2 * fc_normalized))
        hd[n != 0] = -2 * fc_normalized * (np.sin(n[n != 0] * wc) / (n[n != 0] * wc))

    elif filter_type == "Band Pass":
        fc_normalized = (cutoff_freq - (transition_band / 2)) / sampling_freq
        f2_normalized = (f2 + (transition_band / 2)) / sampling_freq
        # Angular frequencies
        wc = 2 * np.pi * fc_normalized
        w2 = 2 * np.pi * f2_normalized
        # Calculate filter coefficients
        hd[n == 0] = 2 * (f2_normalized - fc_normalized)
        hd[n != 0] = ((2 * f2_normalized) * (np.sin(n[n != 0] * w2) / (n[n != 0] * w2))) - \
                     ((2 * fc_normalized) * (np.sin(n[n != 0] * wc) / (n[n != 0] * wc)))

    elif filter_type == "Band Stop":

        # Calculate normalized cutoff frequencies
        fc_normalized = (cutoff_freq + (transition_band / 2)) / sampling_freq
        f2_normalized = (f2 - (transition_band / 2)) / sampling_freq

        # Angular frequencies
        wc = 2 * np.pi * fc_normalized
        w2 = 2 * np.pi * f2_normalized

        # Calculate filter coefficients
        hd[n == 0] = 1 - 2 * (f2_normalized - fc_normalized)
        hd[n == 0] = 1 - 2 * (f2_normalized - fc_normalized)
        hd[n != 0] = ((2 * fc_normalized) * (np.sin(n[n != 0] * wc) / (n[n != 0] * wc))) - \
                     ((2 * f2_normalized) * (np.sin(n[n != 0] * w2) / (n[n != 0] * w2)))
    else:
        raise ValueError("Invalid filter type.")

    if window_type == "Rectangular":
        w = np.ones_like(n)
    elif window_type == "Hanning":
        w = 0.5 + 0.5 * np.cos(2 * np.pi * n / N)
    elif window_type == "Hamming":
        w = 0.54 + 0.46 * np.cos(2 * np.pi * n / N)
    elif window_type == "Blackman":
        w = 0.42 + 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))

    h = hd * w

    coefficients = [(int(idx), coeff) for idx, coeff in zip(n, h)]
    # Print the filter coefficients
    # print("Filter Coefficients (h[n]):")
    # for idx, coeff in coefficients:
    #     print(f"{idx} {coeff:.9f}")

    # Ask if the user wants to upload an ECG signal
    upload_choice = messagebox.askyesno("Upload ECG File", "Would you like to upload an ECG signal for filtering?")
    if upload_choice:
        ecg_file = filedialog.askopenfilename(
            title="Upload ECG File",
            filetypes=[("Text files", ".txt"), ("All files", ".*")]
        )
        ecg_indices, ecg_samples = read_ecg_file(ecg_file)
        conv_result = np.convolve(ecg_samples, h)
        conv_indices = list(range(len(conv_result)))
        conv_samples = conv_result.tolist()
        Your_indices = [int(idx) for idx, _ in coefficients]  # Extract indices

        expected_file = filedialog.askopenfilename(
            title="Select Expected Output File",
            filetypes=[("Text files", ".txt"), ("All files", ".*")]
        )

        if expected_file:
            Compare_Signals(expected_file, Your_indices, conv_samples)
            # print(conv_samples, conv_indices)
        else:
            print("Output file not provided. Unable to compare results.")
            messagebox.showerror("Error", "Output file not uploaded. Please upload an output file before proceeding.")

        # Plot the filtered signal
        ax_task7.clear()
        ax_task7.plot(conv_indices, conv_samples, label="Filtered Signal", color="blue")
        ax_task7.set_title(f"{filter_type} Filtered Signal")
        ax_task7.set_xlabel("Sample Index")
        ax_task7.set_ylabel("Amplitude")
        ax_task7.grid(True)
        canvas_task7.draw()

    else:
        expected_file = filedialog.askopenfilename(
            title="Select Expected Output File",
            filetypes=[("Text files", ".txt"), ("All files", ".*")]
        )

        if expected_file:  # If a file was selected
            print(f"Selected file: {expected_file}")

            # Now compare your filter output with the expected file
            Your_indices = [int(idx) for idx, _ in coefficients]  # Extract indices
            Your_samples = [coeff for _, coeff in coefficients]  # Extract filter coefficients
            # Call Compare_Signals to compare your output with the expected file
            Compare_Signals(expected_file, Your_indices, Your_samples)
        else:
            print("Output file not provided. Unable to compare results.")
            messagebox.showerror("Error", "Output file not uploaded. Please upload an output file before proceeding.")

        # Plot the filter coefficients
        ax_task7.clear()
        ax_task7.plot(n, h, label='Filter Coefficients', color='blue')
        ax_task7.set_title(f'{filter_type} Filter Coefficients')
        ax_task7.set_xlabel('n')
        ax_task7.set_ylabel('h[n]')
        ax_task7.grid(True)
        canvas_task7.draw()

def convolve(x, y):
    xOfk1 = []
    xOfk2 = []
    yOfk = []
    convsignal = []

    N1 = len(x)
    N2 = len(y)
    new_len = N1 + N2 - 1

    for i in range(new_len - len(x)):
        x.append(0)
    for i in range(new_len - len(y)):
        y.append(0)

    # DFT

    j = 1j
    for k in range(len(x)):
        harmonic_value = 0
        for n in range(len(x)):
            harmonic_value += (x[n] * np.exp((-j * 2 * np.pi * k * n) / len(x)))
        xOfk1.append(harmonic_value)

    for k in range(len(y)):
        harmonic_value = 0
        for n in range(len(y)):
            harmonic_value += (y[n] * np.exp((-j * 2 * np.pi * k * n) / len(y)))
        xOfk2.append(harmonic_value)

        # multiply xOfk1 * xOfk2
    for i in range(len(xOfk1)):
        yOfk.append(xOfk1[i] * xOfk2[i])

        # IDFT
    for n in range(len(yOfk)):
        harmonic_value = 0
        for k in range(len(yOfk)):
            harmonic_value += (1 / len(yOfk)) * (yOfk[k] * np.exp((j * 2 * np.pi * n * k) / len(yOfk)))
        convsignal.append(np.real(harmonic_value))

    return convsignal


def Window_Method(samplingfreq, transition, attenutation):
    global N
    global indices
    indices = []
    w = []

    if (attenutation <= 21):
        N = np.ceil((samplingfreq * 0.9) / transition)
        if (N % 2 == 0):
            N += 1

        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            indices.append(i)
            w.append(1)

    elif (attenutation <= 44):
        N = np.ceil((samplingfreq * 3.1) / transition)
        if (N % 2 == 0):
            N += 1

        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            indices.append(i)
            w.append(0.5 + (0.5 * np.cos((2 * np.pi * i) / N)))

    elif (attenutation <= 53):
        N = np.ceil((samplingfreq * 3.3) / transition)
        if (N % 2 == 0):
            N += 1

        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            indices.append(i)
            w.append(0.54 + (0.46 * np.cos((2 * np.pi * i) / N)))

    elif (attenutation <= 74):
        N = np.ceil((samplingfreq * 5.5) / transition)
        if (N % 2 == 0):
            N += 1

        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            indices.append(i)
            w.append(0.42 + (0.5 * np.cos((2 * np.pi * i) / (N - 1))) + (0.08 * np.cos((4 * np.pi * i) / (N - 1))))

    return w


def low_pass_filter(signal, transition, samplingfreq, attenutation):
    global N
    w = []
    h = []
    filter = []
    result = []

    w = Window_Method(samplingfreq, transition, attenutation)

    # f = int(simpledialog.askstring("Input", "Enter Cut Off Frequency:"))
    M = int(M_entry.get())

    f=int(cutoff_freq_entry.get())

    Fc = (f + (transition / 2)) / samplingfreq
    for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
        if (i == 0):
            h.append(2 * Fc)
            continue
        h.append(2 * Fc * ((np.sin(i * (2 * np.pi * Fc))) / (i * (2 * np.pi * Fc))))

    length = len(w)
    # Multiplication
    for i in range(length):
        filter.append(w[i] * h[i])

    result = convolve(filter, signal)
    return result
def upsampling(signal, L):
    up_sampled = []
    for i in range(len(signal)):
        up_sampled.append(signal[i])
        if i == len(signal) - 1:
            break
        for j in range(L - 1):
            up_sampled.append(0)
    return up_sampled

# Downsampling Function
def downsampling(signal, M):
    return [signal[i] for i in range(0, len(signal), M)]
def resampling():
    # Get inputs from GUI
    global indices,N
    indices = []

    sampling_freq = float(sampling_freq_entry.get())  # Sampling frequency in Hz
    stop_atten = float(stop_atten_entry.get())  # Stop band attenuation (dB)
    transition_band = float(transition_band_entry.get())  # Transition band in Hz
    M = int(M_entry.get())  # Downsampling factor
    L = int(L_entry.get())  # Upsampling factor
    cutoff_freq = float(cutoff_freq_entry.get())  # Cutoff frequency in Hz

    # Load signal file
    # signal_file = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])
    # if not signal_file:
    #     print("No signal file selected.")
    #     return
    upload_choice = messagebox.askyesno("Upload ECG File", "Would you like to upload an ECG signal for filtering?")
    if upload_choice:
        ecg_file = filedialog.askopenfilename(
            title="Upload ECG File",
            filetypes=[("Text files", ".txt"), ("All files", ".*")]
        )
        ecg_indices, ecg_samples = read_ecg_file(ecg_file)

    # Read signal
    # ySignal = []
    # with open(signal_file, 'r') as f:
    #     for i in range(3):  # Skip header lines
    #         next(f)
    #     for line in f:
    #         parts = line.strip().split()
    #         ySignal.append(float(parts[1]))

    # Resampling logic
    if M == 0 and L != 0:
        upsampled_signal = upsampling(ecg_samples, L)
        result = low_pass_filter(upsampled_signal,transition_band, sampling_freq,stop_atten)
    elif M != 0 and L == 0:
        filtered_signal = low_pass_filter(ecg_samples,transition_band, sampling_freq,stop_atten)
        result = downsampling(filtered_signal, M)
    elif M != 0 and L != 0:
        upsampled_signal = upsampling(ecg_samples,L)
        filtered_signal = low_pass_filter(upsampled_signal,transition_band, sampling_freq,stop_atten)
        result = downsampling(filtered_signal, M)
    for i in range(int(((N - 1) / 2) + 1), int(len(result) - ((N - 1) / 2))):
        indices.append(i)


    expected_file_path = filedialog.askopenfilename(
        title="Select Expected Output File",
        filetypes=[("Text Files", "*.txt")]
    )
    if not expected_file_path:
        print("No expected output file selected.")
        return


    # print("Resampling Complete.")
    # print("Resampled Signal:", result)
    Compare_Signals(expected_file_path,indices, result)
    ax_task7.clear()
    ax_task7.plot(indices, result, label='Resampling', color='blue')
    ax_task7.set_xlabel('n')
    ax_task7.set_ylabel('h[n]')
    ax_task7.grid(True)
    canvas_task7.draw()


def Compare_Signals(file_name, Your_indices, Your_samples):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")


# --- GUI Setup ---

root = tk.Tk()
root.title("Signal Processing Tasks")
root.configure(bg='black')

# Notebook for Tabs
notebook = ttk.Notebook(root)
task1_frame = tk.Frame(notebook, bg='black')
task2_frame = tk.Frame(notebook, bg='black')
task3_frame = tk.Frame(notebook, bg='black')
task4_frame = tk.Frame(notebook, bg='black')
task5_frame = tk.Frame(notebook, bg='black')
task6_frame = tk.Frame(notebook, bg='black')
task7_frame = tk.Frame(notebook, bg='black')
notebook.add(task1_frame, text="Task 1")
notebook.add(task2_frame, text="Task 2")
notebook.add(task3_frame, text="Task 3")
notebook.add(task4_frame, text="Task 4")
notebook.add(task5_frame, text="Task 5")
notebook.add(task6_frame, text="Task 6")
notebook.add(task7_frame, text="Task 7")
notebook.pack(fill=tk.BOTH, expand=1)

# --- Task 1 Setup ---
fig_task1, (ax1_task1, ax2_task1) = plt.subplots(2, 1, facecolor='black', figsize=(8, 6))
canvas_task1 = FigureCanvasTkAgg(fig_task1, master=task1_frame)
canvas_task1.get_tk_widget().pack(fill=tk.BOTH, expand=1)

toolbar_task1 = NavigationToolbar2Tk(canvas_task1, task1_frame)
toolbar_task1.update()
canvas_task1.get_tk_widget().pack()

task1_input_frame = tk.Frame(task1_frame, bg='black')
task1_input_frame.pack(pady=20)

tk.Label(task1_input_frame, text="Amplitude:", fg="white", bg="black").grid(row=0, column=0)
amp_entry = tk.Entry(task1_input_frame)
amp_entry.grid(row=0, column=1)

tk.Label(task1_input_frame, text="Sampling Frequency:", fg="white", bg="black").grid(row=1, column=0)
freq_samp_entry = tk.Entry(task1_input_frame)
freq_samp_entry.grid(row=1, column=1)

tk.Label(task1_input_frame, text="Analog Frequency:", fg="white", bg="black").grid(row=2, column=0)
freq_entry = tk.Entry(task1_input_frame)
freq_entry.grid(row=2, column=1)

tk.Label(task1_input_frame, text="Phase Shift:", fg="white", bg="black").grid(row=3, column=0)
phase_entry = tk.Entry(task1_input_frame)
phase_entry.grid(row=3, column=1)

tk.Label(task1_input_frame, text="Signal Type:", fg="white", bg="black").grid(row=4, column=0)
signal_menu = ttk.Combobox(task1_input_frame, values=["Sine", "Cosine"], state="readonly")
signal_menu.set("Sine")
signal_menu.grid(row=4, column=1)

task1_button = tk.Button(task1_frame, text="Plot and Compare Signal", command=task1_update_plot, bg="#3DBBC7",
                         fg="black")
task1_button.pack(pady=10)

# --- Task 2 Setup ---
fig_task2, (ax1_task2, ax2_task2) = plt.subplots(2, 1, facecolor='black', figsize=(8, 6))
canvas_task2 = FigureCanvasTkAgg(fig_task2, master=task2_frame)
canvas_task2.get_tk_widget().pack(fill=tk.BOTH, expand=1)

toolbar_task2 = NavigationToolbar2Tk(canvas_task2, task2_frame)
toolbar_task2.update()
canvas_task2.get_tk_widget().pack()

task2_input_frame = tk.Frame(task2_frame, bg='black')
task2_input_frame.pack(pady=20)

tk.Label(task2_input_frame, text="Choose Operation:", fg="white", bg="black").grid(row=0, column=0, padx=10, pady=5)
operation_menu = ttk.Combobox(task2_input_frame,
                              values=["Addition", "Subtraction", "Multiplication", "Square", "Normalize",
                                      "Accumulation"], state="readonly")
operation_menu.set("Addition")
operation_menu.grid(row=0, column=1)

normalize_var = tk.IntVar()
normalize_frame = tk.Frame(task2_input_frame, bg='black')
normalize_frame.grid(row=1, column=0, columnspan=2, pady=5)

tk.Label(normalize_frame, text="Normalization Direction:", fg="white", bg="black").grid(row=0, column=0, padx=10)
tk.Radiobutton(normalize_frame, text="From 0 to 1", variable=normalize_var, value=1, bg='black', fg='white').grid(row=0,
                                                                                                                  column=1)
tk.Radiobutton(normalize_frame, text="From -1 to 1", variable=normalize_var, value=0, bg='black', fg='white').grid(
    row=0, column=2)

tk.Label(task2_input_frame, text="Constant for Multiplication:", fg="white", bg="black").grid(row=2, column=0, padx=10,
                                                                                              pady=5)
multiplication_entry = tk.Entry(task2_input_frame)
multiplication_entry.grid(row=2, column=1)

task2_button = tk.Button(task2_frame, text="Process and Plot Signal", command=task2_update_plot, bg="#3DBBC7",
                         fg="black")
task2_button.pack(pady=10)

# --- Task 3 Setup ---

# Set up the main frame for Task 3
table_frame = tk.Frame(task3_frame, bg='black')
table_frame.pack(pady=20)

# Label and Entry for number of bits
tk.Label(task3_frame, text="Number of Bits:", fg="white", bg="black").pack()
bits_entry = tk.Entry(task3_frame)
bits_entry.pack()

# Label and Entry for number of levels
tk.Label(task3_frame, text="Number of Levels:", fg="white", bg="black").pack()
levels_entry = tk.Entry(task3_frame)
levels_entry.pack()

# Quantize button to trigger calculation
task3_button = tk.Button(task3_frame, text="Quantize Signal", command=on_calculate_task3, bg="#3DBBC7", fg="black")
task3_button.pack(pady=10)

# Frame for displaying results in a table format
# This frame will display the output table based on the function display_task3_results
table_frame = tk.Frame(task3_frame, bg='black')
table_frame.pack(pady=20)

# --- Task 4 Setup ---
tk.Label(task4_frame, text="Sampling Frequency (Hz):", fg="white", bg="black").pack(pady=(10, 5))
sampling_freq_entry = tk.Entry(task4_frame)
sampling_freq_entry.pack(pady=(0, 10))

calculate_button = tk.Button(task4_frame, text="Calculate DFT/IDFT", command=on_calculate_task4, bg="#3DBBC7",
                             fg="black")
calculate_button.pack(pady=10)

# Frame to hold the plot canvas with two subplots for discrete plots
fig_task4, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
fig_task4.patch.set_facecolor('black')
canvas_task4 = FigureCanvasTkAgg(fig_task4, master=task4_frame)
canvas_task4.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# --- Task 5 Setup ---

# Frame for plotting section
fig_task5, ax_task5 = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
fig_task5.patch.set_facecolor('black')
ax_task5.set_facecolor('white')
canvas_task5 = FigureCanvasTkAgg(fig_task5, master=task5_frame)
canvas_task5.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Toolbar for the plot
toolbar_task5 = NavigationToolbar2Tk(canvas_task5, task5_frame)
toolbar_task5.update()
canvas_task5.get_tk_widget().pack()

# Frame for inputs and operations
task5_controls = tk.Frame(task5_frame, bg='black')
task5_controls.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

# Frequency Domain Label and DCT Controls
tk.Label(task5_controls, text="Frequency Domain", fg="white", bg="black", font=("Arial", 10, "bold")).grid(row=0,
                                                                                                           column=0,
                                                                                                           columnspan=2,
                                                                                                           pady=(10, 5))
tk.Label(task5_controls, text="Number of Coefficients:", fg="white", bg="black").grid(row=1, column=0, padx=5, pady=5,
                                                                                      sticky="w")
coefficients_entry = tk.Entry(task5_controls)
coefficients_entry.grid(row=1, column=1, padx=5, pady=5)
dct_button = tk.Button(task5_controls, text="Calculate DCT", bg="#3DBBC7", fg="black",
                       command=lambda: on_calculate_task5_dct())
dct_button.grid(row=2, column=0, columnspan=2, pady=10)

# Time Domain Label and Buttons
tk.Label(task5_controls, text="Time Domain", fg="white", bg="black", font=("Arial", 10, "bold")).grid(row=3, column=0,
                                                                                                      columnspan=2,
                                                                                                      pady=(20, 5))

# Sharpening Button
tk.Button(task5_controls, text="Sharpening", bg="#3DBBC7", fg="black", command=DerivativeSignal).grid(row=4, column=0,
                                                                                                      columnspan=2,
                                                                                                      pady=(10, 5))

# Delaying/Advancing a Signal
tk.Label(task5_controls, text="Delay Signal (k steps):", fg="white", bg="black").grid(row=5, column=0, padx=5, pady=5,
                                                                                      sticky="w")
delay_k_entry = tk.Entry(task5_controls, width=5)
delay_k_entry.grid(row=5, column=1, padx=5, pady=5, sticky="w")
tk.Button(task5_controls, text="Delay Signal", bg="#3DBBC7", fg="black",
          command=lambda: delay_signal(delay_k_entry)).grid(row=6, column=0, columnspan=2, pady=10)

tk.Label(task5_controls, text="Advance Signal (k steps):", fg="white", bg="black").grid(row=7, column=0, padx=5, pady=5,
                                                                                        sticky="w")
advance_k_entry = tk.Entry(task5_controls, width=5)
advance_k_entry.grid(row=7, column=1, padx=5, pady=5, sticky="w")
tk.Button(task5_controls, text="Advance Signal", bg="#3DBBC7", fg="black",
          command=lambda: advance_signal(advance_k_entry)).grid(row=8, column=0, columnspan=2, pady=10)

# Folding a Signal
tk.Button(task5_controls, text="Folding Signal", bg="#3DBBC7", fg="black",
          command=lambda: on_calculate_task5_folding()).grid(row=9, column=0, columnspan=2, pady=(20, 5))

# Delaying/Advancing a Folded Signal
tk.Label(task5_controls, text="Delay Folded (k steps):", fg="white", bg="black").grid(row=10, column=0, padx=5, pady=5,
                                                                                      sticky="w")
folded_delay_k_entry = tk.Entry(task5_controls, width=5)
folded_delay_k_entry.grid(row=10, column=1, padx=5, pady=5, sticky="w")
tk.Button(task5_controls, text="Delay Folded Signal", bg="#3DBBC7", fg="black",
          command=lambda: on_calculate_task5_pshiftfolding()).grid(row=11, column=0, columnspan=2, pady=10)

tk.Label(task5_controls, text="Advance Folded (k steps):", fg="white", bg="black").grid(row=12, column=0, padx=5,
                                                                                        pady=5, sticky="w")
folded_advance_k_entry = tk.Entry(task5_controls, width=5)
folded_advance_k_entry.grid(row=12, column=1, padx=5, pady=5, sticky="w")
tk.Button(task5_controls, text="Advance Folded Signal", bg="#3DBBC7", fg="black",
          command=lambda: on_calculate_task5_shiftfolding()).grid(row=13, column=0, columnspan=2, pady=10)

# --- Task 6 Setup ---
# Frame for plotting section
fig_task6, ax_task6 = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
fig_task6.patch.set_facecolor('black')
ax_task6.set_facecolor('white')
canvas_task6 = FigureCanvasTkAgg(fig_task6, master=task6_frame)
canvas_task6.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Toolbar for the plot
toolbar_task6 = NavigationToolbar2Tk(canvas_task6, task6_frame)
toolbar_task6.update()
canvas_task6.get_tk_widget().pack()

# Frame for inputs and operations
task6_controls = tk.Frame(task6_frame, bg='black')
task6_controls.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

# Window Size Label and Controls
tk.Label(task6_controls, text="Window Size:", fg="white", bg="black").pack(pady=(10, 5))
window_size_entry = tk.Entry(task6_controls)
window_size_entry.pack(pady=(0, 10))

# Smoothing Button
calculate_button = tk.Button(
    task6_controls,
    text="Smoothing",
    command=lambda: smoothing_signal(window_size_entry.get()),
    bg="#3DBBC7",
    fg="black"
)
calculate_button.pack(pady=10)

# Remove DC Component in Time Domain Button
remove_dc_time_button = tk.Button(
    task6_controls,
    text="Remove DC (Time Domain)",
    command=lambda: calc_dc(),
    bg="#3DBBC7",
    fg="black"
)
remove_dc_time_button.pack(pady=10)
# Remove DC Component in freq Domain Button
remove_dc_freq_button = tk.Button(
    task6_controls,
    text="Remove DC (Freq Domain)",
    command=lambda: remove_dc_frequency_dft_idft(),
    bg="#3DBBC7",
    fg="black"
)
remove_dc_freq_button.pack(pady=10)

# Convolution Button
convolution_button = tk.Button(
    task6_controls,
    text="Convolution",
    command=lambda: convolution_task(),
    bg="#3DBBC7",
    fg="black"
)
convolution_button.pack(pady=10)

# Correlation Button
correlation_button = tk.Button(
    task6_controls,
    text="Correlation",
    command=lambda: calculate_correlation(),
    bg="#3DBBC7",
    fg="black"
)
correlation_button.pack(pady=10)

# --- Task 7 Setup ---

# Frame for plotting section
fig_task7, ax_task7 = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
fig_task7.patch.set_facecolor('black')
ax_task7.set_facecolor('white')
canvas_task7 = FigureCanvasTkAgg(fig_task7, master=task7_frame)
canvas_task7.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Toolbar for the plot
toolbar_task7 = NavigationToolbar2Tk(canvas_task7, task7_frame)
toolbar_task7.update()
canvas_task7.get_tk_widget().pack()

# Frame for inputs and operations
task7_controls = tk.Frame(task7_frame, bg='black')
task7_controls.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

# Filter Type Selection
tk.Label(task7_controls, text="Filter Type:", fg="white", bg="black").pack(pady=(10, 5))
filter_type_menu = ttk.Combobox(
    task7_controls, values=["Low Pass", "High Pass", "Band Pass", "Band Stop"], state="readonly"
)
filter_type_menu.set("Low Pass")
filter_type_menu.pack(pady=(0, 10))

# Sampling Frequency Input
tk.Label(task7_controls, text="Sampling Frequency (Hz):", fg="white", bg="black").pack(pady=(10, 5))
sampling_freq_entry = tk.Entry(task7_controls)
sampling_freq_entry.pack(pady=(0, 10))

# Cutoff Frequencies Input
tk.Label(task7_controls, text="Cutoff Frequency (Hz):", fg="white", bg="black").pack(pady=(10, 5))
cutoff_freq_entry = tk.Entry(task7_controls)
cutoff_freq_entry.pack(pady=(0, 10))

# Second Cutoff Frequency Input for Band Pass/Stop Filters
tk.Label(task7_controls, text="Second Cutoff Frequency (Hz):", fg="white", bg="black").pack(pady=(10, 5))
second_cutoff_freq_entry = tk.Entry(task7_controls)
second_cutoff_freq_entry.pack(pady=(0, 10))

# Stop Band Attenuation Input
tk.Label(task7_controls, text="Stop Band Attenuation (\u03B4s):", fg="white", bg="black").pack(pady=(10, 5))
stop_atten_entry = tk.Entry(task7_controls)
stop_atten_entry.pack(pady=(0, 10))

# Transition Band Input
tk.Label(task7_controls, text="Transition Band (Hz):", fg="white", bg="black").pack(pady=(10, 5))
transition_band_entry = tk.Entry(task7_controls)
transition_band_entry.pack(pady=(0, 10))

tk.Label(task7_controls, text="M:", fg="white", bg="black").pack(pady=(10, 5))
M_entry = tk.Entry(task7_controls)
M_entry.pack(pady=(0, 10))

tk.Label(task7_controls, text="L:", fg="white", bg="black").pack(pady=(10, 5))
L_entry = tk.Entry(task7_controls)
L_entry.pack(pady=(0, 10))



tk.Button(
    task7_controls,
    text="Filter Signal",
    command=lambda: filter_signal_task7(),  # Calls the filter function
    bg="#3DBBC7",
    fg="black"
).pack(pady=10)

tk.Button(
    task7_controls,
    text="Resampling",
    command=lambda: resampling(),
    bg="#3DBBC7",
    fg="black"
).pack(pady=10)


# --- Shared Error Label ---
error_label = tk.Label(root, text="", fg="yellow", bg="black")
error_label.pack(pady=5)

root.mainloop()