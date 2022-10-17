#essential imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
from numpy import sin, cos, pi
from scipy.fftpack import fft, fftfreq, ifft
import random
from streamlit_custom_slider import st_custom_slider
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# page construction
st.title("Sampling Studio For Biological Signals")
st.sidebar.title("Sampling Settings")
st.markdown(" Welcome To Our Sampling Studio ")

#cash using(mini memory for the front end)
@st.cache(persist=True)

#functions implementations used through the sampling studio

#generating a random signal function
def generate_signal(time_domain):
    F1 = random.randint(1, 100)
    F2 = random.randint(1, 100)
    randomly_generated_signal =(2*sin(2*pi*F1*time_domain)) + (4*sin(2*pi*F2*time_domain))
    #Adding Guassian Noise
    randomly_generated_signal += (3*np.random.randn(time_domain.size))
    print('Noise added')
    return randomly_generated_signal

#reading Ekg Data to be plotted function
def load_data(select):
    if select == "EMG Sample Signal":
        column_names = ['emg', 't']
        mvc1 = pd.read_csv('MVC1.txt', sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
    elif select == 'EKG Sample Signal':
        column_names = ['ekg', 't']
        mvc1 = pd.read_csv('MVC1.txt', sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
    elif select =="ECG Sample Signal":
        data = np.loadtxt('ECG.dat',unpack=True)
        mvc1 = pd.DataFrame(data)
        mvc1.columns=['ECG']
    else:
        column_names = ['eeg', 't']
        mvc1 = pd.read_csv('MVC1.txt', sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
    return mvc1

#extractiong maximum frequency from a signal function
# def exract_max_frequency_of_signal(input_signal):

#sampling a signal function
# def signal_sampling(sample_rate):

#reconstruct signal from sampling functions
# def reconstruct_signal():

# #read the file provided by the user and plot the signal funtion
# def open_file():

#Sample Signals for the user to try sampling rate change on
selected_signal = st.sidebar.selectbox('Provided Signals', ['Generate A Random Signal', 'EKG Sample Signal', 'ECG Sample Signal', 'EMG Sample Signal', 'EEG SampleSignal', 'Provide A Local File Signal'], key='1')

# sampling_frequency_value_slider = st.slider('Change The Sampling Frequency', value = maximum_sampling_frequency_slider_value)

frequencies = ['2fmax', '3fmax', '4fmax', '5fmax','6fmax','7fmax','8fmax','9fmax', '10fmax']

user_maximum_sampling_frequency= st.selectbox('User Can Change The Maximum Sampling Frequency From Here', frequencies )
user_maximum_sampling_frequency_position = frequencies.index(user_maximum_sampling_frequency) + 1
# maximum_sampling_frequency_slider_value = extract_max_frequency_of_signal(random_signal)* user_maximum_sampling_frequency_position 
SNR= st_custom_slider('SNR', 0, 10,0,key='SNR')
# Noise function 
def createNoise(SNR,Signal_v ):
    # change signal data to array 
    signal_volt=Signal_v[0:300].to_numpy()
    # calculate power in watto off signal 
    Signal_power=signal_volt**2
    # calculate avarage power of signal
    Signal_avg_power=np.mean(Signal_power)
    # change signal into db
    signal_avg_db = 10 * np.log10(Signal_avg_power)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = signal_avg_db - SNR
    noise_avg_power = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    #Generate random guassian noise
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_power), len(signal_volt))
    #return noisy signal
    return noise_volts+signal_volt
if selected_signal == "Generate A Random Signal":
    Fs=40
    delay_frequecny=0.0780
    N=int(Fs/delay_frequecny)
    Tw=N/Fs
    t=np.linspace(0,Tw,num=N)
    random_signal = generate_signal(t)
    random_signal_dict = {"time": t, "signal": random_signal}
    fig = plt.plot("t","signal", data = random_signal_dict)
    st.pyplot(fig)
elif selected_signal == "EMG Sample Signal":
    emg = load_data( selected_signal )
    fig = px.line(emg.t, emg.emg)
    st.plotly_chart(fig)
elif selected_signal == "ECG Sample Signal":
    ecg = load_data( selected_signal ) 
    if SNR==0 :
        fig = px.line(ecg[0:300])
        st.plotly_chart(fig)
    else:
        noised_signal=createNoise(SNR,ecg)
        noise_fig=px.line(noised_signal)
        st.plotly_chart(noise_fig)
elif selected_signal == "EEG Sample Signal":
    eeg = load_data( selected_signal )
    fig = px.line(eeg.emg, eeg.t)
    st.plotly_chart(fig)
else:
    load_data( selected_signal)
# else:
#     #user will provide its own file08

