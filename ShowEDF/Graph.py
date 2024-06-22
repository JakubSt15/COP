import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

class SignalPlot():
    def __init__(self, layout, data, channels_to_plot):

        self.plotWidget = pg.PlotWidget()
        self.channels = channels_to_plot
        self.numberOfChannels = len(self.channels)
        self.colorList = ['#0081AF', '#EAD2AC', '#EABA6B ', '#9A7AA0', '#F7B2AD', '#E94F37', '#FFFC31', '#5C415D', '#F6F7EB', '#6C809A']
        self.cmap = pg.colormap.get('CET-L17')
        self.maxLen = 5
        self.signalPlotData = self.filter_channels(data, self.channels) # raw EDF file
        
        ''' holders for plotted values '''
        self.currentPlot = {}
        self.plotHandler = None
        self.curveHandlers = []
        self.currentSample = 0
        self.plottedDecimation = 10
        self.plotGap = 0.00004
        self.initPlotHandler()
        layout.addWidget(self.plotWidget)

    ''' Gets from EDF only these channels that we need '''
    def filter_channels(self, data, channels):
        lower_to_original_ch_names = {ch.lower(): ch for ch in data.ch_names}
        return {ch: data[lower_to_original_ch_names[ch.lower()]] for ch in channels if ch.lower() in lower_to_original_ch_names}


    def initPlotHandler(self):
        self.plotWidget = pg.GraphicsLayoutWidget()
        
        self.plotHandler = self.plotWidget.addPlot(row=0, col=0)
        self.plotHandler.setXRange(0, self.maxLen, padding=0.1)
        self.plotHandler.setYRange(-0.00005, 0.0008)
        self.plotHandler.showGrid(x=True, y=True)
        self.plotHandler.setLabel('bottom', 'Time (s)')
        labelsGap = 0.045
        labelsXstart = 0.92
        labelsYstart = 0.85
        for i, channel in enumerate(self.channels):
            color = self.colorList[i % len(self.colorList)]
            curve = self.plotHandler.plot(pen=pg.mkPen(color))
            
            label = pg.LabelItem(channel, color=color)
            label.setParentItem(self.plotHandler)
            label.anchor(itemPos=(0.0, 0.0), parentPos=(labelsXstart, labelsYstart - i*labelsGap))

            new_data = self.signalPlotData[channel]
            dataX = new_data[0][0] + (i * self.plotGap)
            timeY = new_data[1]

            dataX_decimated = dataX[::10]
            timeY_decimated = timeY[::10]

            curve.setData(timeY_decimated, dataX_decimated)
            self.curveHandlers.append(curve)
            self.plotHandler.setDownsampling(True, True, 'subsample')
            self.plotHandler.setClipToView(True)