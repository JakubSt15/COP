import numpy as np
from PyQt5 import QtCore, QtWidgets
import mne
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from datetime import timedelta

from mne import create_info
from mne.io import RawArray

from DoctorMenu import DoctorMenuList

plt.style.use('dark_background')

class Ui_MainWindow(object):

    def __init__(self):
        self.timerStatus = False
        self.signalFrequency = 512
        self.elapsed_time = 0  # Time elapsed when the timer is running
        self.start_time = 0  # Time when the timer starts
        self.timer = QtCore.QTimer()
        self.signalTimer = QtCore.QTimer()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Create buttons without setting geometry
        self.CloseButton = QtWidgets.QPushButton("Close", self.centralwidget)
        self.CloseButton.clicked.connect(self.close_window)

        self.StopButton = QtWidgets.QPushButton("Stop", self.centralwidget)
        self.StopButton.clicked.connect(self.stop_timer)

        self.SaveButton = QtWidgets.QPushButton("Save to EDF", self.centralwidget)
        self.SaveButton.clicked.connect(self.save_to_edf)

        # Set central widget and status bar
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # Create a layout for the central widget
        self.layout = QtWidgets.QVBoxLayout(self.centralwidget)

        self.channels_to_plot = ['EEG Fp1', 'EEG F3', 'EEG C3',
                                 'EEG P3', 'EEG O1', 'EEG F7', 'EEG T3',
                                 'EEG T5', 'EEG Fz', 'EEG Cz', 'EEG Pz',
                                 'EEG Fp2', 'EEG F4', 'EEG C4', 'EEG P4',
                                 'EEG O2', 'EEG F8', 'EEG T4', 'EEG T6']

        # Buffers to store data and times
        self.data_buffers = {channel: ([], []) for channel in self.channels_to_plot}

        self.figure, self.axes = plt.subplots(len(self.channels_to_plot), 1, sharex=True, figsize=(10, 20))
        plt.subplots_adjust(bottom=0.05, left=0.005, top=0.95)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, MainWindow)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # Create a horizontal layout for the buttons
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.buttonLayout.addWidget(self.StopButton)
        self.buttonLayout.addWidget(self.CloseButton)
        self.buttonLayout.addWidget(self.SaveButton)  # Add Save button
        self.layout.addLayout(self.buttonLayout)

        # Slider setup
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 64)
        self.slider.setValue(0)
        self.slider.sliderReleased.connect(self.slider_update_plot)
        self.layout.addWidget(self.slider)

        self.source_current_start_idx = 0
        self.source_current_end_idx = 64
        self.source_initial_range = 512


        self.load_file()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.signalTimer.setInterval(125)
        self.signalTimer.timeout.connect(self.signal_update_time)

        self.timer.setInterval(125)
        self.timer.timeout.connect(self.update_plot)

    def epilepsy_prediction(self, data, frequency):
        dummy_signal = data[:, -frequency:]
        dummy_signal = dummy_signal.T
        model_prediction = np.random.random(dummy_signal.shape)
        model_detection = np.random.randint(0, 2, dummy_signal.shape)
        data_to_return = np.any(data == 1, axis=1).astype(int)
        return np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    def close_window(self):
        QtWidgets.qApp.closeAllWindows()
        self.doctorMenu = QtWidgets.QMainWindow()
        self.secondWindow = DoctorMenuList.Ui_MainWindow()
        self.secondWindow.setupUi(self.doctorMenu)
        self.doctorMenu.show()

    def stop_timer(self):
        _translate = QtCore.QCoreApplication.translate
        if self.timerStatus == False:
            self.timerStatus = True
            self.start_time = QtCore.QTime.currentTime()
            self.timer.start()
            self.signalTimer.start()
            self.StopButton.setText(_translate("MainWindow", "Stop"))
        else:
            self.timerStatus = False
            self.timer.stop()
            self.signalTimer.stop()
            stop_time = QtCore.QTime.currentTime()
            self.elapsed_time += self.start_time.msecsTo(stop_time) / 1000.0
            self.StopButton.setText(_translate("MainWindow", "Start"))

    def load_file(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name = '../PN00-4.edf'

        if file_name:
            self.raw = mne.io.read_raw_edf(file_name, preload=True)

    def signal_update_time(self):
        self.source_initial_range += 64
        self.source_current_start_idx += self.signalFrequency // 8
        self.source_current_end_idx += self.signalFrequency // 8

    def update_plot(self):
        slider_value = self.slider.value()
        self.slider.setRange(0, self.source_initial_range - 64)
        self.slider.setValue(self.source_initial_range - 64)

        for ax in self.axes:
            ax.clear()

        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(len(self.channels_to_plot))]

        current_time = self.elapsed_time + self.start_time.msecsTo(QtCore.QTime.currentTime()) / 1000.0
        x_labels = [current_time - (self.source_initial_range / self.signalFrequency) + (i / self.signalFrequency) for i in range(self.source_initial_range)]

        for i, channel in enumerate(self.channels_to_plot):
            data, times = self.raw[channel, self.source_current_start_idx:self.source_current_end_idx]

            self.data_buffers[channel][0].extend(data[0])
            self.data_buffers[channel][1].extend(x_labels)

            temp = self.epilepsy_prediction(data, self.signalFrequency)
            if temp[i] == 1:
                self.axes[i].plot(x_labels[-64:], data[0], label=channel, color='red')
            else:
                self.axes[i].plot(x_labels[-64:], data[0], label=channel, color=colors[i])

            self.axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            self.axes[i].set_yticklabels([])
            self.axes[i].spines['bottom'].set_visible(False)
            self.axes[i].spines['right'].set_visible(False)
            self.axes[i].spines['left'].set_visible(False)

        self.axes[-1].set_xlabel('Time')
        self.axes[-1].set_xticklabels([str(timedelta(seconds=int(sec))) for sec in x_labels[-64:]])
        self.canvas.draw()

    def slider_update_plot(self):
        slider_value = self.slider.value()
        self.current_start_idx = slider_value
        self.current_end_idx = min(self.source_initial_range, self.current_start_idx + 64)

        for ax in self.axes:
            ax.clear()

        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(len(self.channels_to_plot))]

        buffer_data = {channel: np.array(data) for channel, (data, _) in self.data_buffers.items()}
        buffer_times = {channel: np.array(times) for channel, (_, times) in self.data_buffers.items()}

        for i, channel in enumerate(self.channels_to_plot):
            if len(buffer_data[channel]) > self.current_end_idx:
                data = buffer_data[channel][self.current_start_idx:self.current_end_idx]
                times = buffer_times[channel][self.current_start_idx:self.current_end_idx]

                self.axes[i].plot(times, data, label=channel, color=colors[i])
                self.axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                self.axes[i].set_yticklabels([])
                self.axes[i].spines['bottom'].set_visible(False)
                self.axes[i].spines['right'].set_visible(False)
                self.axes[i].spines['left'].set_visible(False)

        self.axes[-1].set_xlabel('Time')
        self.axes[-1].set_xticklabels([str(timedelta(seconds=int(sec))) for sec in buffer_times[self.channels_to_plot[-1]][self.current_start_idx:self.current_end_idx]])
        self.canvas.draw()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.CloseButton.setText(_translate("MainWindow", "Close"))
        self.StopButton.setText(_translate("MainWindow", "Start"))

    def save_to_edf(self):
        ch_names = self.channels_to_plot
        sfreq = self.signalFrequency
        ch_types = ['eeg'] * len(ch_names)

        data = []
        for channel in ch_names:
            data.append(np.array(self.data_buffers[channel][0]))

        data = np.array(data)

        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = RawArray(data, info)
        raw.save('output.edf', overwrite=True)
        QtWidgets.QMessageBox.information(self.centralwidget, "Save EDF", "Data saved to output.edf")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
