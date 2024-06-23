import pyqtgraph as pg
from collections import deque
from queue import Queue
import numpy as np
import threading
import mne
from PyQt5.QtWidgets import QFileDialog

from CommonTools.CommonTools import show_popup
class SignalPlot():
    def __init__(self, layout, channels_to_plot, data, predictorFunction):

        self.plotWidget = pg.PlotWidget()
        self.channels = channels_to_plot
        self.numberOfChannels = len(self.channels)
        self.colorList = ['#31f766', '#008a25']
        self.cmap = pg.colormap.get('CET-L17')
        self.maxLen = 5

        ''' signalPlotData - przefiltrowane kanały'''
        self.signalPlotData = self.filter_channels(data, self.channels)  # raw EDF file
        self.yRangeSet = False
        self.initialYRange = None

        ''' holders for plotted values '''
        self.currentPlot = {}
        self.q = [Queue() for i in range(self.numberOfChannels)]
        self.dat = [deque() for i in range(self.numberOfChannels)]
        self.timeDeq = deque()
        self.plotHandler = None
        self.curveHandlers = []
        self.currentSample = 1076224//10//2 + (512*25) # przenosi do 21:40 min
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

        ''' table helper data structure '''
        self.table_data_buffers = [{'channelName': channel, 'attackProbability': 0} for channel in self.channels]

    ''' Gets from EDF only these channels that we need '''

    def filter_channels(self, data, channels):
        lower_to_original_ch_names = {ch.lower(): ch for ch in data.ch_names}
        return {ch: data[lower_to_original_ch_names[ch.lower()]] for ch in channels if
                ch.lower() in lower_to_original_ch_names}

    def save_to_edf(self):
        """Saves the buffered EEG data to an EDF file."""

        channel_names = list(self.data_buffers.keys())
        data_values = np.array(list(self.data_buffers.values()))
        data_values = data_values.T
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=512,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(data_values.T, info)

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(None, "Wybierz nazwę pliku", "", "EDF Files (*.edf)",
                                                   options=options)
        if file_path:
            if not file_path.endswith('.edf'):
                file_path += '.edf'
            mne.export.export_raw(file_path, raw, 'edf', overwrite=True)
            show_popup("Zapisano", f"Plik EDF zapisano w: {file_path}")

    def initPlotHandler(self):
        self.plotWidget = pg.GraphicsLayoutWidget()

        self.plotHandler = self.plotWidget.addPlot(row=0, col=0)
        self.plotHandler.setXRange(0, self.maxLen, padding=0.1)
        self.plotHandler.showGrid(x=True, y=True)
        self.plotHandler.setLabel('bottom', 'Time (s)')

        labelsGap = 0.045
        labelsXstart = 0.92
        labelsYstart = 0.85
        for i, channel in enumerate(self.channels):
            color = self.colorList[i % len(self.colorList)]
            curve = self.plotHandler.plot(pen=pg.mkPen(color))
            self.curveHandlers.append(curve)

            label = pg.LabelItem(channel)
            label.setParentItem(self.plotHandler)
            label.anchor(itemPos=(0.0, 0.0), parentPos=(labelsXstart, labelsYstart - i * labelsGap))

        self.plotHandler.getAxis('left').setStyle(tickFont=None, showValues=False)
        self.plotHandler.setMouseEnabled(y=False)
        self.plotHandler.setDownsampling(True, True, 'subsample')
        self.plotHandler.setClipToView(True)

    def clear_plot(self, data_times):
        # Remove old plot item
        if self.plotHandler:
            self.plotWidget.removeItem(self.plotHandler)

            # Reset data, create new plot item` and curves
        self.data_buffers = {ch: [] for ch in self.channels}
        self.elapsed_time = 0
        self.source_current_start_idx = 0
        self.source_current_end_idx = data_times
        self.temp = [0] * len(self.channels)
        self.currentSample = 0
        self.timeDeq.clear()

        self.plotHandler = self.plotWidget.addPlot(row=0, col=0)
        self.plotHandler.setXRange(0, self.maxLen, padding=0.1)
        self.plotHandler.showGrid(x=True, y=True)
        self.plotHandler.setLabel('bottom', 'Time (s)')
        self.curveHandlers = []  # Reset curve handlers

        for i in range(self.numberOfChannels):
            color = self.colorList[i % len(self.colorList)]
            curve = self.plotHandler.plot(pen=pg.mkPen(color))
            self.curveHandlers.append(curve)
            self.dat[i].clear()

    def update(self, followPlot=True):
        ''' signal - Y axis, time - X axis'''
        _new_data = self.signalPlotData['eeg fp1']
        _timeY = _new_data[1]
        plotSamplNumber = self.currentSample * self.plottedDecimation
        currentRecordTime = _timeY[plotSamplNumber]
        attackMeanProba = None
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
                self.data_buffers[channel] = self.signalPlotData[channel][0][0][plotSamplNumber - 512:plotSamplNumber]

            values_list = np.array(list(self.data_buffers.values()))
            prediction = self.predict(values_list.T, predictProba=self.modelPredictsProbability)

            ''' out of (n, 19) calculates (1, 19) vector with mean probability for attack for whole second'''
            attackMeanProba = np.mean(prediction, axis=0)
            self.updatePlotColors(attackMeanProba)

        self.currentSample += 1
        current_time = self.timeDeq[-1]
        if current_time > self.maxLen and followPlot:
            self.plotHandler.setXRange(current_time - self.maxLen, current_time, padding=0.1)

            if not self.yRangeSet:
                self.initialYRange = self.plotHandler.getViewBox().viewRange()[1]
                self.yRangeSet = True
            else:
                self.plotHandler.setYRange(*self.initialYRange, padding=0)
                self.plotHandler.setXRange(current_time - self.maxLen, current_time, padding=0.1)

        return attackMeanProba

    def predict(self, data, predictProba=False):
        return self.predictAttack(data, 512, predictProba=predictProba)

    ''' predictions is a (n, 19) element vector of prorbabilites for attack on each channel'''

    def updatePlotColors(self, predictions):
        for i in range(self.numberOfChannels):
            color = self._getColorBasedOnPrediction(predictions[i], i)
            self.curveHandlers[i].setPen(pg.mkPen(color))

    def _getColorBasedOnPrediction(self, prediction, id):
        if prediction > 0.9:
            return '#fc0303'
        elif prediction > 0.8:
            return '#fc3503'
        elif prediction > 0.7:
            return '#fc6b03'
        elif prediction > 0.6:
            return '#fcb503'
        elif prediction > 0.5:
            return '#fcf403'
        elif prediction > 0.25:
            return '#bafc03'
        else:
            return self.colorList[id % len(self.colorList)]
