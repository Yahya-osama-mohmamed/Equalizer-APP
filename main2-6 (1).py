import sys
import csv
import pandas as pd
import pyqtgraph as pg
from mainwindow2 import Ui_MainWindow
from PyQt5 import QtWidgets,QtCore,QtGui,uic
import numpy as np
from numpy.fft import fft, fftfreq      
from random import randint       
from threading import Timer
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors 
import librosa
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import butter, lfilter
# export PYTHONPATH="$PYTHONPATH:/path_to_myapp/myapp/myapp/"

# from pyqtgraph import *
# from PyQt5.QtWidgets import QFileDialog


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
    
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


        self.channels = [self.ui.channel1, self.ui.channel2, self.ui.channel3, self.ui.channel4, self.ui.channel5]

        self.verticalSliders = [self.ui.verticalSlider, self.ui.verticalSlider_2,
                                self.ui.verticalSlider_3, self.ui.verticalSlider_4,
                                self.ui.verticalSlider_5, self.ui.verticalSlider_6,
                                self.ui.verticalSlider_7,self.ui.verticalSlider_8,
                                self.ui.verticalSlider_9,self.ui.verticalSlider_10
                                ]

        self.spectrogram_objects=[self.ui.label, self.ui.label_2, self.ui.label_3, 
                              self.ui.horizontalSlider, self.ui.horizontalSlider_2,
                              self.ui.comboBox]



        self.setup_sliders()

        
        self.hide_objects()




# -------------------------------------------------------------------------------------
        #initializing global variables
        self.fs = 0 
        self.data = []
        self.delay=10
        self.y=[]

        self.i_1 = 0

        #Zooming
        self.g1 =0
        self.j1 =0
        self.x_range=0

		#Horizontal slider variables 
        self.pos = 0
        self.x_range = 0
        self.x_end = 0


	    #Choosing color to plot input channels with.
        self.pen1 = pg.mkPen(color=(0, 0, 255))

        
		#Connect UI some menu actions and checkBox to their functions.
        self.ui.actionChannel_1.triggered.connect(self.getfiles)
        self.ui.actionNew_Window.triggered.connect(self.new_window)
        self.ui.spectrogram.triggered.connect(self.spect)
        self.ui.generateReport.triggered.connect(self.generate_report)
        self.ui.exit.triggered.connect(exit)



        self.ui.checkBox.clicked.connect(self.flag)

        # self.ui.pushButton.clicked.connect(self.spect)

        self.ui.comboBox.currentIndexChanged.connect(self.spect)
        self.ui.horizontalSlider.valueChanged.connect(self.spect)
        self.ui.horizontalSlider_2.valueChanged.connect(self.spect)




#---------------------------------------------------------------------------------
    def setup_sliders(self):

        for slider in self.verticalSliders:
            slider.hide()
            slider.setTickPosition(QtWidgets.QSlider.TicksRight)
            slider.setMinimum(0)
            slider.setMaximum(5)
            slider.setValue(1)
            slider.setTickInterval(1)
            
            slider.valueChanged.connect(self.changeSlidersValue)


    def hide_objects(self):

        for channel in self.channels:
            channel.hide()

        for object_ in self.spectrogram_objects:
            object_.hide()

        self.ui.slider1.hide()   
        self.ui.checkBox.hide()


   
    def connect_objects(self):

        self.ui.slider1.valueChanged.connect(self.slider_1)
        self.ui.play.triggered.connect(self.timer1.start)
        self.ui.pause.triggered.connect(self.timer1.stop)
        self.ui.delete_2.triggered.connect(self.clear1)
        self.ui.zoomin.triggered.connect(lambda:self.zoom('in'))
        self.ui.zoomout.triggered.connect(lambda:self.zoom('out'))
        self.ui.actionSpeed_Up.triggered.connect(lambda:self.speed_up(self.timer1))
        self.ui.actionSpeed_Down.triggered.connect(lambda:self.slow_down(self.timer1))
        self.ui.actionPlay_Audio.triggered.connect(self.play_audio)
        self.ui.actionPause_Audio.triggered.connect(self.pause_audio)

    def flag(self):
        if self.ui.checkBox.isChecked() == True :
            self.hide_spectrogram()
        else:
            self.show_spectrogram()




#-------------------------------------------------------------------------
    def getfiles(self):
        options =  QtWidgets.QFileDialog.Options()
        fname = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "",
                        "(*.wav)", options=options) 

        if(fname[0]!=''):#if path exists
            self.read_data(fname) 
        else:
            pass 


    def read_data(self,fname):
        self.path = fname[0]
        self.data, self.fs = librosa.load(self.path, sr=None, duration=20.0)
        # print(self.data)
        # print(self.fs)
        self.datafft = fft(self.data)
        self.fftabs = abs(self.datafft)
        self.freqs = fftfreq(len(self.fftabs), 1 / self.fs)
        self.N = len(self.fftabs)
        self.T = int(self.N / self.fs)
        # print(self.T)
        self.t = 1 / self.fs * np.arange(self.N)

        self.input_channels()

        # self.ui.channel1.plot(self.data[:self.T * self.fs], pen='b') 
        
        


        # self.chan(self.data1) 

  
    def input_channels (self):
        for channel in self.channels:
            channel.show()
            
        self.ui.channel3.plot(self.freqs[:int(self.freqs.size / 2)], self.fftabs[:int(self.freqs.size / 2)], pen='b')
       
        


        self.data_line =self.ui.channel1.plot(self.data,pen=self.pen1)
        #self.data

        self.ui.channel1.plotItem.getViewBox().setAutoPan(x=True)
       

        self.timer1= QtCore.QTimer()
        self.timer1.setInterval(self.delay)
        self.timer1.timeout.connect(lambda:self.update_plot_data(self.data_line))
        self.timer1.start()
        
        self.changeSlidersValue()

        # self.changeSlidersValue()


        self.ui.slider1.show()

        self.connect_objects()

 

    def update_plot_data(self,data_line): 
        
        self.y = self.data[0:self.i_1]
        
        self.i_1 = self.i_1 +1 
        data_line.setData(self.y)
        
        if len(self.y)< 1100:
            for channel in [self.ui.channel1, self.ui.channel2]:
                channel.setXRange(0,1100)
            
        else: 
            for channel in [self.ui.channel1, self.ui.channel2]:
                channel.setXRange(len(self.y)-1100,len(self.y))

            self.ui.slider1.setValue(len(self.y))



    def changeSlidersValue(self):

        gainArray = []

        for slider in self.verticalSliders:#to show the sliders
            slider.show()

        for slider in self.verticalSliders:#to get values from the sliders
            gainArray.append(slider.value())

        self.output_signal = self.processAudio(gainArray,1)
        self.output_freq= self.processAudio(gainArray,2)
        self.output_channels()
        self.gain = gainArray

        self.spect()
        
        print('Equilizer Gain Array:{}'.format(gainArray))
       # return Rs
        return gainArray


    # https://stackoverflow.com/questions/54932976/audio-equalizer
    # def processAudio(self, gain1, gain2, gain3, gain4, gain5, gain6, gain7, gain8, gain9, gain10,a):
    #     freq = np.arange(self.fs * 0.5)
    #     size = len(freq) / 10
    #     band1 = self.bandpass_filter(freq[21], freq[int(size)],a, order=4) *10000* gain1 
    #     band2 = self.bandpass_filter(freq[int(size)], freq[2 * int(size)],a, order=4) *10000*gain2 
    #     band3 = self.bandpass_filter(freq[2 * int(size)], freq[3 * int(size)],a, order=4) * 10000*gain3
    #     band4 = self.bandpass_filter(freq[3 * int(size)], freq[4 * int(size)],a, order=4) *10000* gain4
    #     band5 = self.bandpass_filter(freq[4 * int(size)], freq[5 * int(size)],a, order=4) *10000* gain5 
    #     band6 = self.bandpass_filter(freq[5 * int(size)], freq[6 * int(size)],a, order=4) *10000* gain6 
    #     band7 = self.bandpass_filter(freq[6 * int(size)], freq[7 * int(size)],a, order=4) * 10000*gain7 
    #     band8 = self.bandpass_filter(freq[7 * int(size)], freq[8 * int(size)],a, order=4) *10000* gain8 
    #     band9 = self.bandpass_filter(freq[8 * int(size)], freq[9 * int(size)],a, order=4) *10000* gain9 
    #     band10 = self.bandpass_filter(freq[9 * int(size)], freq[-1],a, order=3) *10000* gain10 
    #     osignal = band1 + band2 + band3 + band4 + band5 + band6 + band7 + band8 + band9 + band10
    #     return osignal

    def processAudio(self, gain,a):
        
        band=[]
        # freq = np.arange(440 ,460)
        freq = np.arange(self.fs * 0.5)
        size = len(freq) / 10
        band.append(self.bandpass_filter(freq[21], freq[int(size)],a, order=4) *10000* gain[0])
        for i in range (1,9):
            j=i+1
            band.append(self.bandpass_filter(freq[int(size)*i], freq[int(size)*j],a, order=4) *10000* gain[i])
        band.append(self.bandpass_filter(freq[9 * int(size)], freq[-1],a, order=3) *10000* gain[9])

        return sum(band)
        


    def bandpass_filter(self, lowcut, highcut,a, order=5):
        maxfreq = 0.5 * self.fs 
        low = lowcut / maxfreq
        high = highcut / maxfreq
        c1, c2 = butter(order, [low, high], btype='band', analog=False)
        if a== 1:
        
            filtered = lfilter(c1, c2, self.y)
            return filtered
        if a== 2:
            filtered = lfilter(c1, c2, self.data)
            return filtered



    def output_channels(self):#output       
        for channel in [self.ui.channel2, self.ui.channel4]:
            channel.show()
            channel.clear()
        
        datafftafter = fft(self.output_freq)
        self.fftabsafter = abs(datafftafter)
        self.freqsa = fftfreq(len(datafftafter), 1 / self.fs)
        
        self.ui.channel4.plot(self.freqsa[:int(self.freqsa.size / 2)], self.fftabsafter[:int(self.freqsa.size / 2)], pen='r')
        # self.ui.channel4.setXRange(0,1000)
          
        self.ui.channel2.plotItem.getViewBox().setAutoPan(x=True)
        self.data_line2 =self.ui.channel2.plot(self.output_signal,pen='r')
        
        self.timer1.timeout.connect(lambda:self.update_plot_data(self.data_line2))

        self.N1 = len(self.output_signal)
        self.T1 = int(self.N1 / self.fs)

        # sd.play(self.output_signal, self.fs)
    

         
    # def zoomin1 (self):
    #     self.j1=100
    #     self.g1 = self.g1 + 100
        
    #     if self.pos==0:
    #         self.x_range= len(self.y)- 1000 + self.g1
    #       # for channel in self.channels:
    #         for channel in [self.ui.channel1, self.ui.channel2]:
            
    #          # if channel == self.ui.channel5:
    #          #    pass
    #          # else:
    #             # channel.setXRange(self.x_range-self.j1,len(self.y))
    #             channel.setXRange(self.x_range,len(self.y))

    #     elif self.pos!=0:
    #         self.x_range= self.pos-1000+self.g1
    #         for channel in [self.ui.channel1, self.ui.channel2]:
    #             # if channel == self.ui.channel5:
    #             #     pass
    #             # else:
    #             channel.setXRange(self.pos-1000+self.g1,self.pos+1000)
                    
        



        
                
          
    # def zoomout1 (self):
    #     self.g1=100
    #     self.j1 = self.j1+100
    #     if self.pos==0:
    #         # for channel in self.channels:
    #         for channel in [self.ui.channel1, self.ui.channel2]:
        	  
    #             # if channel == self.ui.channel5:
    #     		   #  pass
    #     	    # else:
    #             channel.setXRange(self.x_range-self.j1,len(self.y))
        
    #     elif self.pos!=0:
    #         # for channel in self.channels:
    #         for channel in [self.ui.channel1, self.ui.channel2]:
            
    #             # if channel == self.ui.channel5:
    #             #     pass
    #             # else:
    #         	channel.setXRange(self.x_range-self.j1,self.pos+1000)
        
    

    def set_xrange(self, left, adj , right):
        for channel in [self.ui.channel1, self.ui.channel2]:
            channel.setXRange(left-adj, right)



    def zoom(self,a):
        if a== 'in':
            self.j1=100
            self.g1 = self.g1 + 100
        
            if self.pos==0:
                self.x_range= len(self.y)- 1000 + self.g1
          
                self.set_xrange(self.x_range, 0 ,len(self.y))
            elif self.pos!=0:
                self.x_range= self.pos-1000+self.g1

                self.set_xrange(self.pos-1000, -self.g1 ,self.pos)

        if a== 'out':
            self.g1=100
            self.j1 = self.j1+100
            if self.pos==0:
            
                self.set_xrange(self.x_range, self.j1 ,len(self.y))

        
            elif self.pos!=0:
           
                self.set_xrange(self.x_range, self.j1 ,self.pos)






    def clear1 (self):
        self.ui.channel1.clear()
        
        self.data = []
        self.i_1 =0   
        self.timer1 = None 
        self.delay =10

        self.hide_objects()
        self.setup_sliders()



    def slider_1(self):
            self.pos= self.ui.slider1.value()
            
            for channel in [self.ui.channel1, self.ui.channel2]:
            # for channel in self.channels:
            #     if channel== self.ui.channel5:
            #         pass
            #     else:
            	channel.setXRange(self.pos,self.pos+1000)


            if len(self.y)<1000:
                self.ui.slider1.setMaximum(len(self.y))    
                
            else:
                self.ui.slider1.setMaximum(len(self.y)-1000)

            self.ui.slider1.setMinimum(0)
    


    def hide_spectrogram(self):
        self.ui.channel5.hide()
        for object_ in self.spectrogram_objects:
            object_.hide()


	

    def show_spectrogram(self):

        self.ui.channel5.show()
        for object_ in self.spectrogram_objects:
            object_.show()




    def setup_horizontalSliders(self):


        for slider in [self.ui.horizontalSlider,self.ui.horizontalSlider_2]:
            if self.dBS.min() <-100000:
                slider.setMaximum(40)
                slider.setMinimum(-121)
            else: 
                slider.setMaximum(self.dBS.max())
                slider.setMinimum(self.dBS.min())                

        if self.ui.horizontalSlider.value() ==0 and self.ui.horizontalSlider_2.value() == 0:
            self.ui.horizontalSlider.setValue(self.dBS.min())
            self.ui.horizontalSlider_2.setValue(self.dBS.max())
            self.spect()

        else:    
            self.ui.horizontalSlider.setValue(self.ui.horizontalSlider.value())
            self.ui.horizontalSlider_2.setValue(self.ui.horizontalSlider_2.value())


    def spect(self):

        axs,fig=self.plot_spect()

    
        fig.savefig('spect1.png')

        img = pg.QtGui.QGraphicsPixmapItem(pg.QtGui.QPixmap("spect1.png"))
    
        self.ui.channel5.addItem(img)

        img.scale(100, -100)

     
        self.ui.checkBox.show()
        self.show_spectrogram()

      
    

    def plot_spect(self):
            
            fig, axs = plt.subplots(1, constrained_layout=True)
            
            f, t, Sxx = signal.spectrogram(self.output_freq, self.fs)
            self.dBS = 10 * np.log10(Sxx)

            self.setup_horizontalSliders()
            
            pcm= axs.pcolormesh(t, f,self.dBS
               ,norm =colors.Normalize(vmin=self.ui.horizontalSlider.value(), vmax=self.ui.horizontalSlider_2.value()),
                shading='gouraud',cmap=self.ui.comboBox.currentText())
            
            fig.colorbar(pcm, ax=axs, extend='max')
            
            
            print('dBS Min:{}'.format(self.dBS.min()))
            print('dBS Max:{}'.format(self.dBS.max()))
            print('Set Vmin::{}'.format(self.ui.horizontalSlider.value()))
            print('Set Vmax:{}'.format(self.ui.horizontalSlider_2.value()))            
            
            self.current_cmap = self.ui.comboBox.currentText()


            axs.set_ylabel('Frequency [Hz]')
            axs.set_xlabel('Time [sec]')                       
            # axs.set_title('spec 1')
            return axs,fig



    def report_page(self, freqs, fftabs, data, xlim):
            fig,axs= plt.subplots(2, constrained_layout=True)
            axs[0].plot(freqs[:int(freqs.size / 2)], fftabs[:int(freqs.size / 2)])
            axs[1].plot(data)
            axs[0].axis(xmin=0,xmax=(xlim * len(self.data)))
            axs[1].axis(xmin=0,xmax=(xlim * len(self.data)))
            return axs, fig

    def generate_report(self):
        
        pp = PdfPages('Report.pdf')
        
    



        if self.data != []:

            # fig1,axs1= plt.subplots(2, constrained_layout=True)
            # axs1[0].plot(self.freqs[:int(self.freqs.size / 2)], self.fftabs[:int(self.freqs.size / 2)])
            # axs1[1].plot(self.data)
            # axs1[0].axis(xmin=0,xmax=(0.01 * len(self.data)))
            # axs1[1].axis(xmin=0,xmax=(0.01 * len(self.data)))
            # fig1.savefig(pp, format='pdf')


            axs1, fig1 = self.report_page(self.freqs,self.fftabs, self.data, xlim = 0.01)
            fig1.savefig(pp, format='pdf')

            # fig, axs = plt.subplots(2, constrained_layout=True)
            # axs[0].plot(self.freqsa[:int(self.freqsa.size / 2)], self.fftabsafter[:int(self.freqsa.size / 2)])
            # axs[1].plot(self.output_signal)
            # axs[0].axis(xmin=0,xmax=(0.055 * len(self.data)))
            # axs[1].axis(xmin=0,xmax=(0.055* len(self.data)))
            # fig.savefig(pp, format='pdf')

            axs, fig = self.report_page(self.freqsa,self.fftabsafter, self.output_signal, xlim = 0.02)
            fig.savefig(pp, format='pdf')


            
            self.plot_spect()

        
            plt.savefig(pp, format='pdf')


        
        
        pp.close()
        import os
        os.startfile('Report.pdf')
        

    def new_window(self):
        self.newWindow = MyWindow()
        self.newWindow.show()

    
    def speed_up(self, timer):
        self.delay = self.delay - self.delay/4
        timer.setInterval(self.delay)

    def slow_down(self, timer):
        self.delay = self.delay + self.delay/2
        timer.setInterval(self.delay)

    def play_audio(self):
        sd.play(self.output_signal, self.fs)

    def pause_audio(self):
        sd.stop()



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow() 
    window.show()
    sys.exit(app.exec_())