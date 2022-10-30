################################## Essential imports ######################################################
from matplotlib.ft2font import HORIZONTAL
from matplotlib.pyplot import margins
from soupsieve import select
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.express as px
from numpy import sin, pi
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
################################## Page Layouts ######################################################
st.set_page_config(
    page_title="Sampling Dashboard",
    page_icon="✅",
    layout="wide",
)
################################## Page construction ######################################################
#Adding css file to webpage

with open("design.css")as f:
    st.markdown(f"<style>{f.read() }</style>",unsafe_allow_html=True)
col_upload,col_add_u=st.columns((2,1))
#col1, col2 = st.columns(2)
#graph3,graph4=st.columns((3,1),gap='small')
select_b, graph2 = st.columns((8, 27))
#col_select, col_delete = st.columns([3,1])
with graph2:
        genre = st.radio(
        "Choose Sampling Method",
        ('Sampling Frequency', 'Multiples of Maximum Frequency'),horizontal=True)
def set_slider(max_range):
            if calculate_max_frequency()==0:
                Nyquist_rate=1
            else:
                Nyquist_rate=calculate_max_frequency()*2
            with graph2:
                if genre == 'Sampling Frequency':
                    user_selected_sampling_frequency = st.slider('Change Sampling Frequency ', 1,max_range,value=int(Nyquist_rate),key='sampling_frequency')
                else:
                    user_selected_sampling_frequency=st.slider("sampling with max frequency multiples", calculate_max_frequency(), 10*calculate_max_frequency(), step = calculate_max_frequency(),key='sampling_frequency2' )
                return user_selected_sampling_frequency
        


################################## Adding variables to session ######################################################
if 'list_of_signals' not in st.session_state:
    st.session_state['list_of_signals']=[]
    st.session_state['sum_of_signals']=np.zeros(1000)
    st.session_state['sum_of_signals_clean']=np.zeros(1000)
    st.session_state['interpolated_signal']=np.zeros(1000)
    st.session_state['resampled_time']=np.zeros(1000)
    st.session_state['figure']=go.Figure()
time = np.linspace(0, 5, 1000)
################################## global variables  ######################################################
#cash using(mini memory for the front end)
@st.cache(persist=True)
################################## Function implementation  ######################################################
################################################################################################################################################
#Read and load data to be plotted function
def load_data(select, uploaded_file=None):
    # if select == "EMG Sample Signal":
    #     column_names = ['time', 'values']
    #     returned_signal = pd.read_csv('EMG.csv', sep = ',', names = column_names, skiprows= 50, skipfooter = 50)
    if select =="Provide A Local File Signal":
        column_names = ['time','value','frequency','amplitude','phase']
        returned_signal = pd.read_csv(uploaded_file, sep = ',', names = column_names,header=0)
    return returned_signal
################################################################################################################################################
# Noise function 
def signal_sampling_with_input_frequency(input_signal, sampling_frequency):
    min_time = np.min(input_signal['time'])
    max_time = np.max(input_signal['time'])
    sampled_signal_time_domain = np.arange(min_time, max_time, 0.25/sampling_frequency)
    sampled_signal_points =np.sin( 2*np.pi* sampling_frequency* sampled_signal_time_domain)
    st.session_state.resampled_time=sampled_signal_time_domain
    return sampled_signal_points, sampled_signal_time_domain

def signal_sampling_with_max_frequency_multiples(input_signal = st.session_state.sum_of_signals):
    min_time = np.min(time)
    max_time = np.max(time)
    max_sampling_frequency_multiple = calculate_max_frequency()
    sampled_signal_time_domain = np.arange(min_time, max_time, 0.25/max_sampling_frequency_multiple)
    sampled_signal_points =np.sin( 2*np.pi* max_sampling_frequency_multiple* sampled_signal_time_domain)
    st.session_state.resampled_time=sampled_signal_time_domain
    return sampled_signal_points, sampled_signal_time_domain
################################################################################################################################################
# interpolating function with sinc
def sinc_interp(input_signal, sampling_time):
    original_signal_amplitude = input_signal['values']
    original_signal_time_domain = input_signal['time']
    if len(original_signal_amplitude) != len(original_signal_time_domain):
        raise Exception
    # Find the period
    Time_period = original_signal_time_domain[1] - original_signal_time_domain[0]
    sincM = np.tile(sampling_time, (len(original_signal_time_domain), 1)) - np.tile(original_signal_time_domain[:, np.newaxis], (1, len(sampling_time)))
    resampled_signal = np.dot(original_signal_amplitude, np.sinc(sincM/Time_period))
    st.session_state.interpolated_signal= resampled_signal
    return resampled_signal
################################################################################################################################################
# Noise function
def createNoise(SNR,signal_input ):
    Signal_volt=signal_input['values']
    # calculate power in watto off signal
    Signal_power=Signal_volt**2
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
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_power), len(Signal_volt))
    #return noisy signal
    return noise_volts+Signal_volt, signal_input['time']
################################################################################################################################################
#function used to clear all ploted sine signals
def clear_data():
    #assign all values to zero
    st.session_state['list_of_signals']=[]
    st.session_state['sum_of_signals']=np.zeros(1000)
    st.session_state['sum_of_signals_clean']=np.zeros(1000)
    st.session_state['interpolated_signal']=np.zeros(1000)
    st.session_state['resampled_time']=np.zeros(1000)
    st.session_state['figure']=go.Figure()
################################################################################################################################################
#function used to delete signals from the plot
def delete(index_to_delete):
    #get the values that make up the signal to be deleted from list then delete it
    delete_frequency= st.session_state.list_of_signals[index_to_delete][0] #get frequency of  signal to be deleted
    delete_amplitude=st.session_state.list_of_signals[index_to_delete][1]#get frequency of  signal to be deleted
    delete_phase=st.session_state.list_of_signals[index_to_delete][2]#get phase of  signal to be deleted
    st.session_state.list_of_signals.pop(index_to_delete)#remove the ddesired signal parameter from session state
    removed_signal = delete_amplitude * sin(2 * pi * delete_frequency* time + delete_phase)
    st.session_state.sum_of_signals-=removed_signal
    #if there's no remaing signals clear the data
    if not st.session_state.list_of_signals:
        clear_data()
#################################################################################################################################################   
#adding the noise function
def noise_sine():
    #if the value of the SNR is zero then there is no noise in the signal 
    if hasattr(st.session_state, "noise_slider_key"):
        if st.session_state.noise_slider_key==0 :
            st.session_state.sum_of_signals=st.session_state.sum_of_signals_clean
            sine_signal_dataFrame=pd.DataFrame(data = [np.array(time),np.array(st.session_state.sum_of_signals)]).T
            sine_signal_dataFrame.columns=['time','values']
            if genre == "Sampling Frequency":
                sampled_signal_points, sampled_signal_time_domain = signal_sampling_with_input_frequency(sine_signal_dataFrame, st.session_state.sampling_frequency)
         
            else : 
                sampled_signal_points, sampled_signal_time_domain = signal_sampling_with_input_frequency(sine_noise_dataFrame, st.session_state.sampling_frequency2)
            st.session_state.interpolated_signal= sinc_interp(sine_signal_dataFrame, sampled_signal_time_domain)       
        else:
            #add noise to summation of signals
            noise_signal_dataFrame=pd.DataFrame(data = [np.array(time),np.array(st.session_state.sum_of_signals_clean)]).T
            noise_signal_dataFrame.columns=['time','values']
            st.session_state.sum_of_signals,time_noise_domain=createNoise(st.session_state.noise_slider_key, noise_signal_dataFrame ) 
            sine_noise_dataFrame=pd.DataFrame(data = [np.array(time_noise_domain),np.array(st.session_state.sum_of_signals)]).T
            sine_noise_dataFrame.columns=['time','values']
            if genre == "Sampling Frequency":
                sampled_signal_points, sampled_signal_time_domain = signal_sampling_with_input_frequency(sine_noise_dataFrame, st.session_state.sampling_frequency)
            else : 
                sampled_signal_points, sampled_signal_time_domain = signal_sampling_with_input_frequency(sine_noise_dataFrame, st.session_state.sampling_frequency2)

            st.session_state.interpolated_signal= sinc_interp(sine_noise_dataFrame, sampled_signal_time_domain)
        
        
######################################################################################################################################################
#converting the the signals indvidually and the sum of the signals into a csv file function
def convert_data_to_csv():
    #dataframe of the phases, frequencies and amplitudes of the signals
    df_of_signals=pd.DataFrame(st.session_state.list_of_signals,columns=['Frequency','Amplitude','Phase'])
    #dataframe of the sum of the signals
    df_sum_signals=pd.DataFrame({"Time": time, "Value": st.session_state.sum_of_signals})
    #add both of the data frames and the sum together horizontally
    csv_file=pd.concat([df_sum_signals,df_of_signals],axis=1)
    return csv_file.to_csv(index=False)
######################################################################################################################################################
def calculate_max_frequency():
    frequencies=[]
    for i in range(len(st.session_state.list_of_signals)):
        frequencies.append(st.session_state.list_of_signals[i][0])
    if frequencies:
        return max(frequencies)
    else:
        return 0
################################## Ploting functions ######################################################
################################################################################################################################################
def add_plot(fig=st.session_state.figure,x=time):
    with select_b:
        signal_sum = st.checkbox('Signals Sum',value=True)
        signal_resampled = st.checkbox('Resampled Signal',value=True)
    
    st.session_state.figure=go.Figure()
    #y_axis=[st.session_state.sum_of_signals,st.session_state.interpolated_signal]
    fig.add_trace(
        go.Scatter(x=x, y=st.session_state.sum_of_signals, name='sum signals',visible=signal_sum)
        ) 
    fig.add_trace(
        go.Scatter(x=st.session_state.resampled_time, y=st.session_state.interpolated_signal, name='Resampled Signal',mode='markers+lines',visible=signal_resampled)
        ) 
    signals=st.session_state.list_of_signals
    for i in range(len(st.session_state.list_of_signals)):
        with select_b:
            signal_no = st.checkbox(f'Signal {i+1}')
        #y_axis.append(signals[i])
        y=signals[i][1] * sin(2 * pi * signals[i][0] * time + signals[i][2])
        fig.add_trace(
            go.Scatter(x=x, y=y, name=f'Signal {i+1}',visible=signal_no)
            ) 

def update_plot(fig=st.session_state.figure):
    # updating plot layout by changing color ,adding titles to plot ....
    fig.update_layout(
    xaxis_title="Time (Sec)", 
    yaxis_title="Amplitude (Volt)",
    paper_bgcolor="white",
    font_color="black",
)
    #ploting wave using plotly
    with graph2:
        st.plotly_chart(fig,use_container_width=True)


def edit_sine():
        if st.session_state.list_of_signals:
            delete(-1)
        generate_sine()
        
def generate_sine():
        sine_volt = st.session_state.Amplitude * sin(2 * pi * st.session_state.Frequency * time + st.session_state.Phase)
        #add the signal to the cache storage
        signal_parameters=[st.session_state.Frequency,st.session_state.Amplitude,st.session_state.Phase]
        st.session_state.list_of_signals.append(signal_parameters)
        st.session_state.sum_of_signals+=sine_volt
        st.session_state.sum_of_signals_clean=st.session_state.sum_of_signals
        noise_sine()
def delete_sine():
    if st.session_state.list_of_signals:
        with st.sidebar:
            # col_sel,col_del=st.columns([27,25])
            option = st.selectbox(
            'Select Values to Delete',
            st.session_state.list_of_signals,format_func=lambda x: "Frequency:" + str(x[0])+", Amplitude:" + str(x[1])+", Phase:" + str(x[2]))
            selected_value=st.session_state.list_of_signals.index(option)

                #if the button is pressed go the delete function
            st.button('Delete',key=1,on_click=delete,args= (selected_value,))
        
        #after every change from the upload, delete or genrate we update both plots
def add_sampling_sine():
    sine_signal_dataFrame=pd.DataFrame(data = [np.array(time),np.array(st.session_state.sum_of_signals)]).T
    sine_signal_dataFrame.columns=['time','values']
    sampled_signal_points, sampled_signal_time_domain = signal_sampling_with_input_frequency(sine_signal_dataFrame, sampling_frequecny_applied)
    interpolated_signal= sinc_interp(sine_signal_dataFrame, sampled_signal_time_domain)

################################## Main implementation ######################################################
with st.sidebar:
    #slider to get frequency for sin wave generation
    frequency = st.slider('Frequency', 0, 20,1 , key='Frequency',on_change=edit_sine)
    #slider to get amplitude for sin wave generation
    amplitude = st.slider('Amplitude', 0, 20,1, key='Amplitude',on_change=edit_sine)
    #slider to get phase for sin wave generation
    phase = st.slider('Phase', 0.0, 2*pi,value=0.79, key='Phase',on_change=edit_sine)
    if not st.session_state.list_of_signals:
        generate_sine()
    st.button('Add',on_click=generate_sine)
    # st.session_state.sampling_frequency=st.slider("sampling with max frequency multiples", calculate_max_frequency(), 10*calculate_max_frequency(), step = calculate_max_frequency() )
    sampling_frequecny_applied = set_slider(80)
    noise_sin=st.sidebar.slider('SNR',0,80,0,key="noise_slider_key",on_change=noise_sine,help='0 is equivalent to not having any noise')  

        
#UPLOADING A GENRATED FILE
with col_upload:
    uploaded_file = st.file_uploader("", accept_multiple_files=False,type=['csv','txt'])
with col_add_u:
    add_upload=st.button('Add file')
    #if there's a file uploaded and the button is pressed
if uploaded_file and add_upload :
        #download the data to the browser
        data=load_data( 'Provide A Local File Signal',uploaded_file)
        #get the signal parameters and remove the empty entries
        amplitudes_of_downloaded_signal=np.array(data.amplitude.dropna())
        frequencies_of_downloaded_signal=np.array(data.frequency.dropna())
        phases_of_downloaded_signal=np.array(data.phase.dropna())
        t=np.array(data.time)
        #loop through the present values of the frequencies
        for i in range(len(frequencies_of_downloaded_signal)):
            #calculate the sine and draw it 
            sine_volt = amplitudes_of_downloaded_signal[i] * sin(2 * pi * frequencies_of_downloaded_signal[i] * t + phases_of_downloaded_signal[i])
            
            #add the parameters to the stored data 
            signal_parameters=[frequencies_of_downloaded_signal[i],amplitudes_of_downloaded_signal[i],phases_of_downloaded_signal[i]]
            st.session_state.list_of_signals.append(signal_parameters)
    #add the values(y-axis) to the stored sum
        st.session_state.sum_of_signals+=data.value
        st.session_state.sum_of_signals_clean=st.session_state.sum_of_signals

#if the slider of the noise changes then go noise func
#with graph2:

add_sampling_sine()
delete_sine() 
with st.sidebar:
    but_cle,but,but_save=st.columns([27,1,26])
    but_cle.button("Clear",on_click=clear_data)

    but_save.download_button(
        label="Save ",
        data=convert_data_to_csv(),
        file_name='Sample.csv',
        mime='text/csv',
    )
add_plot()
update_plot()
