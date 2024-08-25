import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib


def Tx_gen(range_max, c, range_resolution, Nd, Nr, fc):
    """
    Generates FMCW Tx signal
    range_max: Maximum range of the target
    c: Speed of light
    range_resolution: FMCW radar range resolution
    Nd: Number of chirps
    Nr: Numnber samples per chirp
    fc: FMCW radar carrier frequency
    
    # Usage Example:
        # Radar parameters settings:    
            range_max = 200 # Maximum range of the target
            c = 3e8 # Speed of light
            range_resolution = 1 # FMCW radar range resolution
            Nd = 64 # Number of chirps
            Nr = 256 # Numnber samples per chirp
            fc = 77e9 # FMCW radar carrier frequency
        Function:
            Tx, t, slope, Tchirp = Tx_gen(range_max, c, range_resolution, Nd, Nr, fc)
        
    """
    
    # Tx Signal
    B = c/(2*range_resolution) # Bandwidth of the signal
    t_chirp = 5.5*2*range_max/c # Chirp signal duration
    slope = B/t_chirp # Slope of the Chirp signal
    
    
    t = np.linspace(0,Nd*t_chirp,Nr*Nd) # Time instances (x-axis vector) of the Tx and Rx signals
    chirp_freq = fc*t+(slope*t*t)/2 # Chirp instantaneous frequency (used in Tx signal equation)
    freq = fc + slope*t # Frequencies of the Tx chirp for each time instance (function of chirp slope)
    Tx = np.cos(2*np.pi*chirp_freq) # Simulated Tx signal vector
    
    return Tx, t, slope, t_chirp


def Rx_gen(Nd, Nr, Tx, endle_time, target_range, target_velocity, t, slope, c, fc, alpha=1):
    """
    Generates FMCW Rx return signal
    Nd: Number of chirps
    Nr: Numnber samples per chirp
    Tx: Transmitted signal
    endle_time: Time delay between ramps
    target_range: Arbitrary target range
    target_velocity: Arbitrary target velocity
    t: A list consisting of time instances (Recommended to use the output from Tx_gen function)
    slope: Slope of the Chirp signal
    c: Speed of light
    fc: FMCW radar carrier frequency
    alpha: Received signal amplitude
    # Usage Example:
        
        target_range = 55 # Arbitrary target range
        target_velocity = 35 # Arbitrary target velocity
        endle_time = 6.3e-6 # Time delay between ramps
        Rx_gen(Nd, Nr, Tx, endle_time, target_range, target_velocity, t, slope, c, fc)
        
    """
    
    
    # Generating Rx and IF signals
    Rx_Signal = np.empty(Nd*Nr) # An empty vector for required dimensions to store Rx signal
    for i in range(len(Tx)):
        endle_adj = -1*endle_time # Time delay between ramps. Starts with a negative value which is compensated in a couple of lines below for the first ramp
        if (i%Nd == 0):
            endle_adj = endle_adj + endle_time # Endle adjustment. Starts with a value of 0 and endle_time is added at the end of each ramp

        r_instant = target_range - target_velocity*(t[i]+endle_adj) # Instantaneous range for a given velocity of the target
        t_delay = 2*r_instant/c # Time delay of the Rx signal due to target range
        freqRx = fc + slope*(t[i]) # Instantaneous slope of the Rx ramp 
        Rx = alpha*np.cos(2*np.pi*(fc*(t[i]-t_delay) + (slope*(t[i]-t_delay)*(t[i]-t_delay))/2)) # Rx signal


        inter_fq = Rx*Tx[i].conjugate() # Rx signal component mixed with the Rx signal to get Intermediate Frequency (IF)

        Rx_Signal[i] = inter_fq # Assigning the intermediate frequency component to the Rx signal vector


    Rx_Signal = Rx_Signal.reshape(Nd,Nr) # Reshape Rx signal to distinguish between ramps
    
    return Rx_Signal


def Rx_mpt_gen(Nd, Nr, Tx, endle_time, target_range, target_velocity, t, slope, c, fc, shape=False, shape_dim=0):
    """
    Generates FMCW Rx return signal from a multi-point target
    Nd: Number of chirps
    Nr: Numnber samples per chirp
    Tx: Transmitted signal
    endle_time: Time delay between ramps
    target_range: Arbitrary target range
    target_velocity: Arbitrary target velocity
    t: A list consisting of time instances (Recommended to use the output from Tx_gen function)
    slope: Slope of the Chirp signal
    c: Speed of light
    fc: FMCW radar carrier frequency
    alpha: Received signal amplitude
    shape: Target shape to simulate. Use 'Square' or 'Triangle'
    shape_dim: Dimensions of the selected shape
    
    # Usage Example:
    
        target_range = 55 # Arbitrary target range
        target_velocity = 35 # Arbitrary target velocity
        endle_time = 6.3e-6 # Time delay between ramps
        Rx_Signals_mpt_sqr = Rx_mpt_gen(Nd, Nr, Tx, endle_time, target_range, target_velocity, t, slope, c, fc, shape="Square", shape_dim=7.5)
        Rx_Signals_mpt_trig = Rx_mpt_gen(Nd, Nr, Tx, endle_time, target_range, target_velocity, t, slope, c, fc, shape="Triangle", shape_dim=7.5)
    
    """
    
    if shape == "Square":
        mpt_alpha0 = [1, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25] # Relative target return amplitudes for the selected shape
        mpt_alpha = [x * 2 for x in mpt_alpha0] # Target return amplitudes for the selected shape
    elif shape == "Triangle":
        mpt_alpha0 = [1, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25] # Relative target return amplitudes for the selected shape
        mpt_alpha = [x * 0.25 for x in mpt_alpha0] # Target return amplitudes for the selected shape
    else:
        mpt_alpha = [1] # Default target return amplitude
    
    # Generating Rx and IF signals
    Rx_signal = np.empty((len(mpt_alpha), Nd*Nr)) # An empty vector for required dimensions to store Rx signal
    for i in range(len(Tx)):
        endle_adj = -1*endle_time # Time delay between ramps. Starts with a negative value which is compensated in a couple of lines below for the first ramp
        if (i%Nd == 0):
            endle_adj = endle_adj + endle_time # Endle adjustment. Starts with a value of 0 and endle_time is added at the end of each ramp

        r_instant = target_range - target_velocity*(t[i]+endle_adj) # Instantaneous range for a given velocity of the target
        
        if shape == "Square": # Multi point reflective parameters for 'Square' target
            r1 = ((shape_dim**2)/16 + r_instant**2)**0.5 # Calculating relative range
            r2 = ((shape_dim**2)/2 + r_instant**2)**0.5 # Calculating relative range           
            
            mpt_range = [r_instant, r1, r1, r1, r1, r2, r2, r2, r2] # Relative multi point ranges
            
            for j in range(len(mpt_alpha)):
                t_delay = 2*mpt_range[j]/c # Time delay of the Rx signal due to target range
                freqRx = fc + slope*(t[i]) # Instantaneous slope of the Rx ramp 
                Rx = mpt_alpha[j]*np.cos(2*np.pi*(fc*(t[i]-t_delay) + (slope*(t[i]-t_delay)*(t[i]-t_delay))/2)) # Rx signal
                
                inter_fq = Rx*Tx[i].conjugate() # Rx signal component mixed with the Rx signal to get Intermediate Frequency (IF)
                
                Rx_signal[j,i] = inter_fq # Assigning the intermediate frequency component to the Rx signal vector
                
        elif shape == "Triangle":
            r1 = ((shape_dim**2)/12 + r_instant**2)**0.5 # Calculating relative range 
            r2 = ((shape_dim**2)/3 + r_instant**2)**0.5 # Calculating relative range            
            
            mpt_range = [r_instant, r1, r1, r1, r2, r2, r2] # Relative multi point ranges
            
            for j in range(len(mpt_alpha)):
                t_delay = 2*mpt_range[j]/c # Time delay of the Rx signal due to target range
                freqRx = fc + slope*(t[i]) # Instantaneous slope of the Rx ramp 
                Rx = mpt_alpha[j]*np.cos(2*np.pi*(fc*(t[i]-t_delay) + (slope*(t[i]-t_delay)*(t[i]-t_delay))/2)) # Rx signal
                
                inter_fq = Rx*Tx[i].conjugate() # Rx signal component mixed with the Rx signal to get Intermediate Frequency (IF)
                
                Rx_signal[j,i] = inter_fq # Assigning the intermediate frequency component to the Rx signal vector
    
    Rx_Signal = np.sum(Rx_signal, axis=0) # Adding individual reflective component returns of the multi point target return
    Rx_Signal = Rx_Signal.reshape(Nd,Nr) # Reshape Rx signal to distinguish between ramps
    
    return Rx_Signal



def Rx_add_noise(Rx_data, SNR_db):
    """
    Adds noise with a required signal-to-noise ratio
    Rx_data: Received signal data#
    SNR_db: Required SNR
    
    # Usage Example:
    
        Rx_Signals_mpt_sqr_noise = Rx_add_noise(Rx_Signals_mpt_sqr, 1)
        
    """
    
    SNR_linear_val = 10 ** (SNR_db / 10)
    
    Rx_dims = Rx_data.shape # Dimensions of the Rx data
    Rx_noisy_signal = np.empty((Rx_dims[0],Rx_dims[1])) # Creating an empty noise vector with the same dimensions as the Rx data
    
    for i in range(Rx_dims[0]): # Adding noise to each ramp
        Rx_signal_lp = np.mean(np.abs(Rx_data[i,:]) ** 2) # Calculating linear power of the Rx data
        RX_noise_lp = Rx_signal_lp/SNR_linear_val # Calculating noise linear power to achieve the required signal-to-noise ratio
        
        Rx_noise_vec = np.sqrt(RX_noise_lp) * np.random.randn(len(Rx_data[i,:])) # Generating a random gaussian noise vector with the required linear power
        
        Rx_noisy_signal[i,:] = Rx_data[i,:] + Rx_noise_vec # Adding noise to each ramp
        
    
    return Rx_noisy_signal


def FFT_gen(Rx_data, Fs, Nr, c, slope, range_window, doppler_window, range_window_en=False, doppler_window_en=False, range_plot=False, doppler_fft=True, doppler_plot_linear=False, doppler_plot_log=False, doppler_data_format='Linear', range_data_format='Linear'):
    """
    Generates range or doppler FFT outputs
    Rx_data: Received data for FFT input
    Fs: Sampling rate
    Nr: Numnber samples per chirp
    c: Speed of light
    slope: Slope of the Chirp signal
    range_window: Range windowing function
    doppler_window: Doppler windowing function
    range_window_en: Boolean. Enables windowing across the range axis
    doppler_window_en: Boolean. Enables windowing across the doppler axis
    range_plot: Boolean. Enables range FFT plot
    doppler_fft: Boolean. If True, doppler FFT data is generated. If False, range FFT data is generated
    doppler_plot_linear: Boolean. Enables linear-doppler FFT plot
    doppler_plot_log: Boolean. Enables log-doppler FFT plot
    doppler_data_format: Format of the doppler FFT data. Select "Linear" or "Log"
    range_data_format: Format of the range FFT data. Select "Linear" or "Log"
    
    # Usage Example:
    
        Fs = Nr/Tchirp # Sampling rate
        range_window = np.hamming(Nr) # hamming window for range FFT data
        doppler_window = np.hamming(Nd) # hamming window for doppler FFT data

        FFT_gen(Rx_Signals_mpt_sqr_noise, Fs, Nr, c, slope, range_window, doppler_window, range_window_en=True, 
                doppler_window_en=True, range_plot=False, doppler_fft=True, doppler_plot_linear=False, 
                doppler_plot_log=False, doppler_data_format='Linear', range_data_format='Linear')
        
    """
    if range_window_en: # Checking if windowing needs to performed on the range axis
        Rx_data = Rx_data * range_window # Applying windowing function (input: range_window) on range axis
        
    fft_data = np.fft.fft(Rx_data, axis=1) # Generating the range FFT data
    fft_range_linear = np.abs(fft_data) # Generating absolute linear values of range FFT
    fft_range_log = 10*np.log10(np.abs(fft_data)) # Generating log values of range FFT
    frequency = np.fft.fftfreq(Nr, 1/Fs) # FFT frequency values
    frequency_range = frequency*c/(2*slope) # FFT frequency values
        
    if range_plot: # Plotting log range FFT
        plt.figure(figsize=(10, 5)) 
        min_xt = np.min(frequency_range)
        max_xt = np.max(frequency_range)
        plt.plot(frequency_range[0:128],fft_range_log[0][0:128]);
    
    if doppler_fft: # Checking if doppler FFT is required
        if doppler_window_en: # Checking if windowing needs to performed on the doppler axis
            fft_data = (fft_data.T * doppler_window).T # Applying windowing function (input: doppler_window) on doppler axis
            
        fft_doppler_linear = np.abs(np.fft.fft(fft_data, axis=0)) # Generating absolute linear values of doppler FFT
        fft_doppler_log = 10*np.log10(fft_doppler_linear) # Generating log values of doppler FFT
        
        if doppler_plot_linear: # Plotting linear doppler FFT
            plt.figure(figsize=(10, 5)) 
            yticklabels = np.round(np.linspace(120, 1, 32),0)
            sns_plot = sns.heatmap(fft_doppler_linear[32:,0:128],yticklabels=yticklabels)
            # sns_plot.invert_yaxis()
        
        if doppler_plot_log: # Plotting log doppler FFT
            plt.figure(figsize=(10, 5)) 
            yticklabels = np.round(np.linspace(120, 1, 32),0)
            sns_plot = sns.heatmap(fft_doppler_log[32:,0:128],yticklabels=yticklabels)
            # sns_plot.invert_yaxis()
            
    if doppler_fft: # Checking if doppler FFT is required
        if doppler_data_format == 'Linear': # Checking if linear doppler FFT is required
            return fft_doppler_linear
        elif doppler_data_format == 'Log': # Checking if log doppler FFT is required
            return fft_doppler_log
    else: # Checking if range FFT is required
        if range_data_format == 'Linear': # Checking if linear range FFT is required
            return fft_range_linear
        elif range_data_format == 'Log': # Checking if log range FFT is required
            return fft_range_log


def Get_Target_Params(Shape, data_points):
    """
    Generates target parameters for a given shape
    
    # Shape: "Square" or "Triangle"
    # data_points: Use the output data set size
    
    # Usage Example:
    
        Get_Target_Params("Square", 5)
    """
    
    
    if Shape == "Square": # Defining the ranges of values that are randomised for the 'Square' shape
        min_range = 1 # Range limits
        max_range = 120 # Range limits
        min_velocity = 35 # Velocity limits
        max_velocity = 40 # Velocity limits
        
        min_shape_dim = 9 # Square dimension limits
        max_shape_dim = 11 # Square dimension limits

        ranges = np.round([random.uniform(min_range, max_range) for _ in range(data_points)],2) # Genearting randomised ranges withing the required limits
        velocities = np.round([random.uniform(min_velocity, max_velocity) for _ in range(data_points)],2) # Genearting randomised velocities withing the required limits
        shape_dims = np.round([random.uniform(min_shape_dim, max_shape_dim) for _ in range(data_points)],2) # Genearting randomised shape dimensions withing the required limits
        
    elif Shape == "Triangle": # Defining the ranges of values that are randomised for the 'Triangle' shape
        min_range = 1 # Range limits
        max_range = 60 # Range limits
        min_velocity = 5 # Velocity limits
        max_velocity = 8 # Velocity limits
        
        min_shape_dim = 6 # Triangle dimension limits
        max_shape_dim = 7 # Triangle dimension limits

        ranges = np.round([random.uniform(min_range, max_range) for _ in range(data_points)],2) # Genearting randomised ranges withing the required limits
        velocities = np.round([random.uniform(min_velocity, max_velocity) for _ in range(data_points)],2) # Genearting randomised velocities withing the required limits
        shape_dims = np.round([random.uniform(min_shape_dim, max_shape_dim) for _ in range(data_points)],2) # Genearting randomised shape dimensions withing the required limits
        
    return [min_range, max_range, min_velocity, max_velocity, 
            min_shape_dim, max_shape_dim, ranges, velocities, shape_dims] 


def Target_Data_gen(Shapes, data_points, Tx, t, slope, Tchirp, Fs, Nd, Nr, endle_time, c, fc, 
                    range_window, doppler_window, SNR_dB, relative_file_path, file_name_suffix="", 
                    Output="Doppler_FFT", range_window_en=True, doppler_window_en=True,
                    range_plot=False, doppler_plot_linear=False, doppler_plot_log=False,
                    write_to_file=False, local_output=False, debug=False):
    """
    Simulates FMCW radar multi-point target return data, performs pre-processing to generate intermediate frequency data, adds gaussian noise with required SNR and produces range or doppler FFT data
    Shapes: List of target shapes. Use "Square", "Triangle"
    data_points: Number of measurement samples to simulate
    Tx: Transmitted signal
    t: A list consisting of time instances ( Recommended to use the output from Tx_gen function )
    slope: Slope of the Chirp signal
    Tchirp: Duration of each chirp
    Fs: Sampling rate
    Nd: Number of chirps
    Nr: Numnber samples per chirp
    endle_time: Time delay between ramps
    c: Speed of light
    fc: FMCW radar carrier frequency 
    range_window: Range windowing function
    doppler_window: Doppler windowing function
    SNR_dB: Required SNR
    relative_file_path: Relative path to the directory where the simulated data is written
    file_name_suffix: Optional. Adds this suffix to the target directory
    Output: Output data format. Select one of "Doppler_FFT", "Range_FFT" and "Time_Domain_Signal"
    range_window_en: Boolean. Enables windowing across the range axis
    doppler_window_en: Boolean. Enables windowing across the doppler axis
    range_plot: Boolean. Enables range FFT plot
    doppler_plot_linear: Boolean. Enables linear - doppler FFT plot
    doppler_plot_log : Boolean. Enables log - doppler FFT plot
    write_to_file: Boolean. If True, writes simulated data to the specified directory location
    local_output: Boolean. If True, returns output within the python environment
    debugBoolean. If True, returns debug data
    
    # Usage Example:
    
    Shapes = ["Square", "Triangle"]
    data_points = 10
    SNR_dB = 3
    relative_file_path = 'data/'
    Target_Data_gen(Shapes, data_points, Tx, t, slope, Tchirp, Fs, Nd, Nr, endle_time, c, fc, 
                    range_window, doppler_window, SNR_dB, relative_file_path, file_name_suffix="", 
                    Output="Doppler_FFT", range_window_en=True, doppler_window_en=True,
                    range_plot=False, doppler_plot_linear=False, doppler_plot_log=False,
                    write_to_file=True, local_output=False, debug=False)
                        
    """
    if local_output: # Checking if local output in python environment is required
        target_data = [] # Creating an empty list to store data for local output
        
    for Shape in Shapes: # Running the loop for each shape (Square and Triangle)
        [min_range, max_range, min_velocity, max_velocity, 
         min_shape_dim, max_shape_dim, ranges, velocities, shape_dims] = Get_Target_Params(Shape, data_points) # Generating target parameters for each shape
        
        file_naming_sequence = 0 # First instance of the file name (starts with '0000' and increments by 1 for each successive data point)
        for i in range(data_points): # Generating one output data sample at a time
            Rx_Signals_mpt_sqr = Rx_mpt_gen(Nd, Nr, Tx, endle_time, ranges[i], 
                                            velocities[i], t, slope, c, fc, shape=Shape, shape_dim=shape_dims[i]) # Generating FMCW Rx multi-point target return signal
            Rx_Signals_mpt_sqr_noise = Rx_add_noise(Rx_Signals_mpt_sqr, SNR_dB) # Adding noise with a required signal-to-noise ratio
            if Output == "Doppler_FFT":
                if i == 0:
                    Output_Data = FFT_gen(Rx_Signals_mpt_sqr_noise, Fs, Nr, c, slope, 
                                                           range_window, doppler_window, range_window_en=True, 
                                                           doppler_window_en=True, range_plot=range_plot, doppler_fft=True, 
                                                           doppler_plot_linear=doppler_plot_linear, doppler_plot_log=doppler_plot_log, 
                                                           doppler_data_format='Linear', range_data_format='Linear') # Generating range or doppler FFT outputs
                else:
                    Output_Data = FFT_gen(Rx_Signals_mpt_sqr_noise, Fs, Nr, c, slope, 
                                                           range_window, doppler_window, range_window_en=True, 
                                                           doppler_window_en=True, range_plot=False, doppler_fft=True, 
                                                           doppler_plot_linear=False, doppler_plot_log=False, 
                                                           doppler_data_format='Linear', range_data_format='Linear') # Generating range or doppler FFT outputs
            elif Output == "Range_FFT":
                if i == 0:
                    Output_Data = FFT_gen(Rx_Signals_mpt_sqr_noise, Fs, Nr, c, slope, 
                                                       range_window, doppler_window, range_window_en=True, 
                                                       doppler_window_en=True, range_plot=range_plot, doppler_fft=False, 
                                                       doppler_plot_linear=doppler_plot_linear, doppler_plot_log=doppler_plot_log, 
                                                       doppler_data_format='Linear', range_data_format='Linear') # Generating range or doppler FFT outputs
                else:
                    Output_Data = FFT_gen(Rx_Signals_mpt_sqr_noise, Fs, Nr, c, slope, 
                                                       range_window, doppler_window, range_window_en=True, 
                                                       doppler_window_en=True, range_plot=False, doppler_fft=False, 
                                                       doppler_plot_linear=False, doppler_plot_log=False, 
                                                       doppler_data_format='Linear', range_data_format='Linear') # Generating range or doppler FFT outputs
               
            elif Output == "Time_Domain_Signal":
                Output_Data = Rx_Signals_mpt_sqr_noise # Time domain noisy multi-point target return signal data output

            if write_to_file: # Checking if output needs to be writted to a local directory
                filename = f"{file_naming_sequence:04}" # First instance of the file name (converting number to a string in four character format: '0000')
                directory_path = relative_file_path + '{}'.format(Shape) + file_name_suffix # Path to the local directory
                if not os.path.exists(directory_path): # Checking if the target local directory does not exist
                    os.mkdir(directory_path) # Creating the target directory if it doesn't exist
                file_path = relative_file_path + '{}'.format(Shape) + file_name_suffix + '/' + '{}'.format(Shape.lower()) + '-' + filename + '.npy' # Generating output directory path to save the data locally
                np.save(file_path, Output_Data.T) # Saving the data in the required local directory
            
            if local_output: # Checking if local output is required
                target_data.append(Output_Data.T) # Appending the data to the local output list 
            
            file_naming_sequence += 1 # Incrementing the file name sequence by 1
            print("--- {}: {}% complete! ---".format(Shape, int((file_naming_sequence)/data_points*100)),end="\r") # Printing status
                
    
            if debug: # Debug mode: Returns sample data for data observability
                return Output_Data
            
    if local_output:
        return np.array(target_data) # Returning local data to python environment


def ML_Model_Tester(Shapes_dict, data_samples, SNR_dB, Output, c, Nd, Nr, fc, Tx, t, slope, Fs, 
                    endle_time, range_window, doppler_window, ML_path, Categorical_y_test=False):
    
    Shapes = list(Shapes_dict.keys()) # Extracting target shape names
    Shapes_list = Shapes*data_samples # Generating required number of target shape sampels
    random.shuffle(Shapes_list) # Shuffling the list of target shape names for randomisation
    target_classifications = [Shapes_dict[key] for key in Shapes_list] # Generating equivalent randomised target classifications
    
    target_data = [] # An empty list to store FMCW radar data
    for Shape in Shapes_list: # Iterating for each shape in randomised Shapes list
        [min_range, max_range, min_velocity, max_velocity, min_shape_dim, 
         max_shape_dim, ranges, 
         velocities, shape_dims] = Get_Target_Params(Shape, 1000) # Generating target parameters for each shape
        
        target_range = random.choice(ranges) # Randomising input target range for the current shape
        target_velocity = random.choice(velocities) # Randomising input target velocity for the current shape
        target_shape_dims = random.choice(shape_dims) # Randomising input target shape dimensions for the current shape
        
        Rx_Signals_mpt_sqr = Rx_mpt_gen(Nd, Nr, Tx, endle_time, target_range, 
                                        target_velocity, t, slope, c, fc, shape=Shape, 
                                        shape_dim=target_shape_dims) # Genearting FMCW Rx multi-point target return signal
        Rx_Signals_mpt_sqr_noise = Rx_add_noise(Rx_Signals_mpt_sqr, SNR_dB) # Adding noise with a required signal-to-noise ratio
        
        if Output == "Doppler_FFT":
            Output_Data = FFT_gen(Rx_Signals_mpt_sqr_noise, Fs, Nr, c, slope, 
                                  range_window, doppler_window, range_window_en=True,
                                  doppler_window_en=True, range_plot=False, doppler_fft=True, 
                                  doppler_plot_linear=False, doppler_plot_log=False, 
                                  doppler_data_format='Linear', range_data_format='Linear') # Generating range or doppler FFT outputs
        
        elif Output == "Range_FFT":
            Output_Data = FFT_gen(Rx_Signals_mpt_sqr_noise, Fs, Nr, c, slope,
                                  range_window, doppler_window, range_window_en=True, 
                                  doppler_window_en=True, range_plot=range_plot, doppler_fft=False, 
                                  doppler_plot_linear=False, doppler_plot_log=False, 
                                  doppler_data_format='Linear', range_data_format='Linear') # Generating range or doppler FFT outputs
                
        elif Output == "Time_Domain_Signal":
            Output_Data = Rx_Signals_mpt_sqr_noise # Time domain noisy multi-point target return signal data output


        target_data.append(Output_Data.T) # Apending FMCW radar data for each iteration of target shape
        
    y_test = np.array(target_classifications) # Converting target classification list to an array
    if Categorical_y_test: # Checking if y_test input needs to be of categorical format
        y_test = to_categorical(y_test, len(Shapes_dict)) # Converting y_test to categorical
    X_test = np.array(target_data) # Converting radar data to an array
    
    ML_model = joblib.load(ML_path) # Importing the previously trained machine learning model from the target directory
    loss, accuracy = ML_model.evaluate(X_test, y_test) # Evaluating the machine learning model using the randomised data
    
    return loss, accuracy


        


