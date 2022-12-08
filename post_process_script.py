import datetime
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib import animation as animation
from random import random, seed
config = {
    'data_collection': {
        'num_sessions': 15,
        'num_measurements': 2,
        'len_sessions': 20,
        'save_dir': 'C:\\deep_orientation_data',
        'calib_fname': 'calib.txt',
        'gt_fname': 'gt.txt',
        'num_frames': 700,
        'discard_first': 200,
        'rad_data_dir': 'C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\',
        'rad_fname': 'adc_data',
        'start_times_fname': 'rad_start_times.txt'
    },

    'format': {
        'num_adc_samples': 256,
        'num_tx_antennas': 2,
        'num_rx_antennas': 4,
        'num_loops': 255,
        'num_range_bins': 512,
        'num_doppl_bins': 64,
        'num_angle_bins': 128,
        'iq': 2,
        'fps': 25,
        'range_res': 0.048
    },

    'sync': {
        'envs': ['env4']
    }
}

RADAR_CUBE_SIZE = config['format']['num_tx_antennas'] * \
                  config['format']['num_rx_antennas'] * \
                  config['format']['num_adc_samples'] * \
                  config['format']['num_loops'] * \
                  config['format']['iq']

rad_data_dir = config['data_collection']['rad_data_dir']
rad_fname = config['data_collection']['rad_fname']

# Name of file where raw data is saved.
data_fname = f'{rad_data_dir}{rad_fname}.bin'
# Name of file that logs measurement start times.
log_fname = f'{rad_data_dir}{rad_fname}_LogFile.txt'


def save_start_time(save_dir, session):
    """
    Append a new line for the radar start time in a text file for 
    the current session.

    This file is used later in sync_gt.py to sync frames with orientation
    annotations.

    -- Inputs:
    save_dir (str): Location where the dataset is saved.
    session  (str): The name of the current measurement (e.g., 1_1, 2_1 etc.)
    """
    save_fname = os.path.join(save_dir, config['data_collection']['start_times_fname'])

    # Read last line of log_fname and parse tstart
    with open(log_fname, 'r') as fh:
        lines = fh.readlines()
        for line in reversed(lines):
            if 'API:SensorStart,0,' in line:
                start_time = line.split(': API:SensorStart,0,')[0].split(' ')[1]
                break

    start_time += '.0'

    with open(save_fname, 'a') as fh:
        fh.write(f'Session: {session}, Start: {start_time}\n')


def load_rad_data(fname=data_fname):
    """
    From the raw data file saved by mmWaveStudio, load raw 
    data frames.

    -- Inputs:
    data_fname (str) (optional): Name of the file where raw data is saved. 
    """
    rad_data = np.fromfile(fname, dtype=np.uint16)
    # Convert from two's complement unsigned int to signed integer decimal representation
    rad_data = rad_data.astype(np.int16)
    rad_data = rad_data.reshape((-1, RADAR_CUBE_SIZE))

    return rad_data


def save_rad_data(save_dir, session):
    """
    Save radar data.

    -- Inputs:
    save_dir (str): Location where the dataset is saved.
    session  (str): The name of the current measurement (e.g., 1_1, 2_1 etc.)    
    """

    # Save measurement start time
    save_start_time(save_dir, session)

    # Load rad data
    rad_data = load_rad_data()

    # Save radar data
    with open(os.path.join(save_dir, f'radar_{session}'), 'wb') as fh:
        fh.write(rad_data[config['data_collection']['discard_first']:, :])
    print('Saved radar data.')


if __name__ == '__main__':
    """ Testing. """
    print("In main")
    data_fname = f'{rad_data_dir}adc_data.bin'
    adc_data = load_rad_data(fname=data_fname)
    final_spectograms=[]
    fig=plt.figure()
    # Iterating through each frame in data bin file
    for frame in range(adc_data.shape[0]):

        frame_real = adc_data[frame, :].reshape(-1, 4)[:, :2].reshape(-1)
        frame_imag = adc_data[frame, :].reshape(-1, 4)[:, -2:].reshape(-1)
        frame_data = frame_real + 1j * frame_imag

        frame_data = frame_data.reshape((config['format']['num_loops'],
                                         config['format']['num_tx_antennas'],
                                         config['format']['num_rx_antennas'],
                                         config['format']['num_adc_samples']))
        avg=np.mean(frame_data, axis=0, keepdims=1)
        background_subtracted_frame_data=frame_data-avg
        print("Print background_subtracted_frame_data_shape "+str(background_subtracted_frame_data.shape))
        range_profile_background_subtracted=np.fft.fft(background_subtracted_frame_data,axis=-1)
        print("Print range_profile_background_subtracted_shape "+str(range_profile_background_subtracted.shape))
        #range_profile=np.fft.fft(frame_data,axis=-1)
        #print("Print range_profile_shape "+str(background_subtracted_frame_data.shape))
        vel_spectogram=np.fft.fft(range_profile_background_subtracted, axis=0, n=256)
        vel_spectogram=np.fft.fftshift(vel_spectogram, axes=0)
        print("Print vel_spectogram_shape "+str(vel_spectogram.shape))
        #final_spectogram=plt.imshow(np.abs(vel_spectogram[128 - 32:128 + 32, 0, 0, :64]))
        mean_tx_axis = np.mean(np.abs(vel_spectogram), axis=1)
        mean_rx_axis = np.mean(mean_tx_axis, axis=1)
        print("Print mean_tx_axis "+str(mean_tx_axis.shape))
        print("Print mean_rx_axis "+str(mean_rx_axis.shape))        
        f = r"D://Wifi_Board_Data_Collection/PC/7_feet/animation_random="+str(random())+".jpeg"
        plt.savefig(f)
        final_spectogram=plt.imshow(np.abs(vel_spectogram[128 - 32:128 + 32, 0, 0, :64]))
        #final_spectogram = plt.imshow(mean_rx_axis[128-32:128+32, :64])

        final_spectograms.append([final_spectogram])
        # print(range_profile.shape)
        # print(range_profile_background_subtracted.shape)
        # plt.subplot(2,1,1)
        # plt.plot(np.abs(range_profile[0,0,0,:]))
        # plt.subplot(2,1,2)
        # plt.plot(np.abs(range_profile_background_subtracted[0,0,0,:]))
        # plt.show()
        # exit()
    # ani = animation.ArtistAnimation(fig, final_spectograms, interval=50, blit=True, repeat_delay=1000)
    # f = r"D://Wifi_Board_Data_Collection/Laptop/1_feet/animation_random="+str(random())+".gif"
    # ani.save(f)
    plt.axhline(32)
    plt.show()