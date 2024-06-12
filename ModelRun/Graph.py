from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot
from collections import deque
from queue import Queue
import numpy as np

class SignalPlot():
    def __init__(self, layout, channels_to_plot, data):

        self.plotWidget = pg.PlotWidget()
        self.channels = channels_to_plot
        self.numberOfChannels = len(channels_to_plot)
        self.colorList = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']
        self.maxLen = 100

        ''' signalPlotData - przefiltrowane kanały'''
        self.signalPlotData = self.filter_channels(data, self.channels) # raw EDF file
        
        ''' holders for plotted values '''
        self.currentPlot = {}
        self.q = [Queue() for i in range(self.numberOfChannels)]
        self.dat = [deque() for i in range(self.numberOfChannels)]
        self.plotHandlers = []
        self.curveHandlers = []
        self.currentSample = 0

        self.initPlotHandler()
        layout.addWidget(self.plotWidget)

        ''' Timer for updating plots '''
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)  # Update interval in milliseconds
        self.timer.timeout.connect(self.update)
        self.timer.start()

    
    ''' Gets from EDF only these channels that we need '''
    def filter_channels(self, data, channels):
        return {ch: data[ch] for ch in channels if ch in data.ch_names}
    
    def initPlotHandler(self):
        self.plotWidget = pg.GraphicsLayoutWidget()
        
        lower_bound_microvolts = 30
        upper_bound_microvolts = 80
        lower_bound_volts = lower_bound_microvolts * 1e-6
        upper_bound_volts = upper_bound_microvolts * 1e-6

        for i in range(self.numberOfChannels):
            plotHandler = self.plotWidget.addPlot(row=i, col=0)
            plotHandler.setYRange(lower_bound_volts, upper_bound_volts, padding=0.02)
            plotHandler.setXRange(0, self.maxLen, padding=0.1)
            plotHandler.showGrid(x=True, y=True)
            color = self.colorList[i % len(self.colorList)]
            curve = plotHandler.plot(pen=pg.mkPen(color))
            self.plotHandlers.append(plotHandler)
            self.curveHandlers.append(curve)
            if i < self.numberOfChannels - 1:
                self.plotWidget.nextRow()
        
        ''' Links all axises together - enables them all to be dragged with mouse at once'''
        for i in range(1, self.numberOfChannels):
            self.plotHandlers[i].setXLink(self.plotHandlers[0])


    def update(self):
        ''' signal - Y axis, time - X axis'''
        # signal, time = self.signalPlotData[:, self.source_current_start_idx:self.source_current_end_idx]

        ''' Update plots '''
        for i, channel in enumerate(self.channels):
            
            new_data = self.signalPlotData[channel]
            dataX = new_data[0][0]
            timeY = new_data[1]

            # if len(self.dat[i]) >= self.maxLen:
                # self.dat[i].popleft()

            self.dat[i].append(dataX[i] * 100000)

            signalY = [x for x in self.dat[i]]
            signalX = list(range(len(signalY)))


            self.curveHandlers[i].setData([self.dat[i][-1], signalY[-1]])

        self.currentSample += 1

        # for i, ax in enumerate(self.axes):
        #     ax.clear()

        #     color = 'red' if self.temp[i] == 1 else plt.cm.tab20(i)
        #     ax.plot(times, data[i], label=self.channels_to_plot[i], color=color)

        #     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #     for spine in ['top', 'bottom', 'right', 'left']:
        #         ax.spines[spine].set_visible(False)

        #     ax.get_yaxis().set_visible(False)

        # self.axes[-1].set_xlabel('Time')
        # self.canvas.draw()