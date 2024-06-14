from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot
from collections import deque
from queue import Queue
import numpy as np

class SignalPlot():
    def __init__(self, layout, channels_to_plot, data, predictorFunction):

        self.plotWidget = pg.PlotWidget()
        self.channels = channels_to_plot
        self.numberOfChannels = len(self.channels)
        self.colorList = ['#31f766', '#008a25']
        self.maxLen = 5

        ''' signalPlotData - przefiltrowane kanały'''
        self.signalPlotData = self.filter_channels(data, self.channels) # raw EDF file
        
        ''' holders for plotted values '''
        self.currentPlot = {}
        self.q = [Queue() for i in range(self.numberOfChannels)]
        self.dat = [deque() for i in range(self.numberOfChannels)]
        self.timeDeq = deque()
        self.plotHandler = None
        self.curveHandlers = []
        self.currentSample = 0
        self.plottedDecimation = 10
        self.initPlotHandler()
        layout.addWidget(self.plotWidget)

        ''' Prediction data '''
        self.predictAttack = predictorFunction
        self.startPredictingSampleSecond = 2
        self.plotGap = 0.00004
        self.lastRecordSecond = 0
        self.modelPredictsProbability = True
        self.data_buffers = {channel: [] for channel in self.channels}


    ''' Gets from EDF only these channels that we need '''
    def filter_channels(self, data, channels):
        return {ch: data[ch] for ch in channels if ch in data.ch_names}
    
    def initPlotHandler(self):
        self.plotWidget = pg.GraphicsLayoutWidget()
        
        self.plotHandler = self.plotWidget.addPlot(row=0, col=0)
        self.plotHandler.setXRange(0, self.maxLen, padding=0.1)
        self.plotHandler.showGrid(x=True, y=True)
        self.plotHandler.setLabel('bottom', 'Time (s)')

        for i in range(self.numberOfChannels):
            color = self.colorList[i % len(self.colorList)]
            curve = self.plotHandler.plot(pen=pg.mkPen(color))
            self.curveHandlers.append(curve)
    
        self.plotHandler.setMouseEnabled(y=False)


    def update(self):
        ''' signal - Y axis, time - X axis'''
        _new_data = self.signalPlotData['EEG Fp1']
        _timeY = _new_data[1]
        plotSamplNumber = self.currentSample * self.plottedDecimation
        currentRecordTime = _timeY[plotSamplNumber]
        self.timeDeq.append(currentRecordTime)

        ''' Update plots '''
        for i, channel in enumerate(self.channels):
            
            new_data = self.signalPlotData[channel]
            dataX = new_data[0][0]
            timeY = new_data[1]
            
            self.dat[i].append(dataX[plotSamplNumber] + (i * self.plotGap))

            self.curveHandlers[i].setData(self.timeDeq, self.dat[i])
            
            


        ''' performing prediction every second '''
        if currentRecordTime > self.startPredictingSampleSecond and currentRecordTime - self.lastRecordSecond > 1:
            self.lastRecordSecond = currentRecordTime
            for i, channel in enumerate(self.channels):
                self.data_buffers[channel] = self.signalPlotData[channel][0][0][plotSamplNumber-512:plotSamplNumber]
            
            values_list = np.array(list(self.data_buffers.values()))
            prediction = self.predict(values_list.T, predictProba=self.modelPredictsProbability)
            print(prediction)

        self.currentSample += 1
        current_time = self.timeDeq[-1]
        if current_time > self.maxLen:
            self.plotHandler.setXRange(current_time - self.maxLen, current_time, padding=0.1)


    def predict(self, data, predictProba=False):
        return self.predictAttack(data, 512, predictProba=predictProba)