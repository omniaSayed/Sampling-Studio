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
from plotly.subplots import make_subplots

# page construction
st.title("Sampling Studio For Biological Signals")
st.sidebar.title("Sampling Settings")
st.markdown(" Welcome To Our Sampling Studio ")
with open("design.css")as f:
    st.markdown(f"<style>{f.read() }</style>",unsafe_allow_html=True)
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

#reading Data to be plotted function
def load_data(select, uploaded_file=None):
    if select == "EMG Sample Signal":
        column_names = ['t', 'emg']
        mvc1 = pd.read_csv('EMG.csv', sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
    elif select == 'EKG Sample Signal':
        column_names = ['t', 'ekg']
        mvc1 = pd.read_csv('MVC1.txt', sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
    elif select =="ECG Sample Signal":
        data = np.loadtxt('ECG.dat',unpack=True)
        mvc1 = pd.DataFrame(data)
        mvc1.columns=['t', 'ECG']
    elif select =="EEG Sample Signal":
        column_names = ['t', 'eeg']
        mvc1 = pd.read_csv('MVC1.txt', sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
    elif select =="Provide A Local File Signal":
        column_names = ['t','value']
        mvc1 = pd.read_csv(uploaded_file, sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
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
selected_signal = st.sidebar.selectbox('Provided Signals', ['Generate A Random Signal', 'EKG Sample Signal', 'ECG Sample Signal', 'EMG Sample Signal', 'EEG Sample Signal', 'Provide A Local File Signal'], key='1')

# sampling_frequency_value_slider = st.slider('Change The Sampling Frequency', value = maximum_sampling_frequency_slider_value)
def set_slider(max_freq):
    st.slider('User Can Change The Maximum Sampling Frequency From Here', 1,3*max_freq )


# maximum_sampling_frequency_slider_value = extract_max_frequency_of_signal(random_signal)* user_maximum_sampling_frequency_position 
with st.sidebar:
   SNR= st_custom_slider('SNR', 0, 20,0,key='SNR')

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
    fig = px.line(x=t, y=random_signal)
    st.plotly_chart(fig)

elif selected_signal == "EMG Sample Signal":
    maximum_frequency=500
    set_slider(maximum_frequency)
    emg = load_data( selected_signal )
    fig = px.line(x=emg.t, y=emg.emg)
    st.plotly_chart(fig)

elif selected_signal == "EKG Sample Signal":
    maximum_frequency=500
    set_slider(maximum_frequency)
    emg = load_data( selected_signal )
    fig = px.line(x=emg.t, y=emg.ekg)
    st.plotly_chart(fig)

elif selected_signal == "ECG Sample Signal":
    maximum_frequency=500
    set_slider(maximum_frequency)
    ecg = load_data( selected_signal ) 
    if SNR==0 :
        fig = px.line(ecg[0:300])
        st.plotly_chart(fig)
    else:
        noised_signal=createNoise(SNR,ecg)
        noise_fig=px.line(noised_signal)
        st.plotly_chart(noise_fig)

elif selected_signal == "EEG Sample Signal":
    maximum_frequency=500
    set_slider(maximum_frequency)
    eeg = load_data( selected_signal )
    fig = px.line(x=eeg.t, y=eeg.eeg)
    st.plotly_chart(fig)

elif selected_signal == 'Provide A Local File Signal':
    uploaded_file = st.file_uploader("Please choose a CSV or TXT file", accept_multiple_files=False,type=['csv','txt'])
    maximum_frequency=500
    set_slider(maximum_frequency)
    if uploaded_file:
        data=load_data( 'Provide A Local File Signal',uploaded_file)
        fig = px.line(x=data.t, y=data.value)
        st.plotly_chart(fig)



# --------------------------------------------------------------------------------------------
SamplingRate = st.slider('sample size', 0, 200, 25)
frequancy = 20
time_step = np.linspace(0, 0.5, 200)
signalWave = np.sin(2*np.pi*frequancy*time_step)
S_rate = SamplingRate

Time = 1/S_rate
num_of_samp = np.arange(0, 0.5/Time)
time_for_sampling = num_of_samp*Time
SignalWave_for_sampling = np.sin(2*np.pi*frequancy*time_for_sampling)






fig= make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Scatter(x=time_step, y=signalWave,name='SineWave of frequency 20 Hz'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=time_for_sampling, y=SignalWave_for_sampling,name='Sample marks after resampling at fs=35Hz', mode='lines+markers'),
    row=1, col=2
)

fig.update_xaxes(title_text='Time', row=1, col=1)
fig.update_xaxes(title_text='Time', row=1, col=2)

fig.update_yaxes(title_text='Amplitude', row=1, col=1)
fig.update_yaxes(title_text='Amplitude', row=1, col=2)

fig.update_layout(height=600, width=800)
st.plotly_chart(fig)
# plt.subplot(2, 2, 2)
# plt.plot(time_for_sampling, SignalWave_for_sampling, 'g-', label='Reconstructed Sine Wave')
# plt.xlabel('time.', fontsize=15)
# plt.ylabel('Amplitude', fontsize=15)
# plt.legend(fontsize=10, loc='upper right')


