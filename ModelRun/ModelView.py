import os
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt, decimate
import torch
from sklearn.metrics import accuracy_score
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QListWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
import matplotlib.pyplot as plt
import mne
from mne import create_info

from DoctorMenu import DoctorMenuList
from Model.prepare_data import prepare_dataset_attack_model, get_attack_sample_from_predictions, \
    prepare_prediction_multi_channel_datasets
from Model.train_attack import AttackModel, MultiChannelAttackModel
from Model.visualize import visualize_predicted_attack
from ModelRun.Graph import SignalPlot

import pyqtgraph as pg

plt.style.use('dark_background')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('FATAL')


class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        self.signalPlot = None
        self.raw = None
        self.setup_initial_values()
        self.setup_timers()
        self.setupUi(MainWindow)

    def setup_initial_values(self):
        mne.set_log_level('CRITICAL')
        self.timerStatus = False
        self.signalFrequency = 512
        self.elapsed_time = 0
        self.start_time = 0
        self.plot_extension = 0
        self.data_times = 32
        self.channels_to_plot = ['EEG Fp1', 'EEG F3', 'EEG C3',
                            'EEG P3', 'EEG O1', 'EEG F7', 'EEG T3',
                            'EEG T5', 'EEG Fz', 'EEG Cz', 'EEG Pz',
                            'EEG Fp2', 'EEG F4', 'EEG C4', 'EEG P4',
                            'EEG O2', 'EEG F8', 'EEG T4', 'EEG T6']
        self.temp = [0] * len(self.channels_to_plot)
        self.source_initial_range = 100
        self.source_current_start_idx = 0
        self.source_current_end_idx = 32

    def setup_timers(self):
        self.timer = QtCore.QTimer()
        self.signalTimer = QtCore.QTimer()
        time = 128
        self.signalTimer.setInterval(time)
        self.signalTimer.timeout.connect(self.signal_update_time)
        self.timer.setInterval(time)
        self.timer.timeout.connect(self.update_plot)

    def setupUi(self, MainWindow):
        self.setup_main_window(MainWindow)
        centralwidget = self.setup_central_widget(MainWindow)
        self.setup_buttons(centralwidget)
        layout = self.setup_layout(centralwidget)
        self.load_file()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def setup_main_window(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        statusbar = QtWidgets.QStatusBar(MainWindow)
        statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(statusbar)

    def setup_central_widget(self, MainWindow):
        centralwidget = QtWidgets.QWidget(MainWindow)
        centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(centralwidget)
        return centralwidget

    def setup_buttons(self, centralwidget):
        self.CloseButton = QtWidgets.QPushButton("Close", centralwidget)
        self.CloseButton.clicked.connect(self.close_window)
        self.StopButton = QtWidgets.QPushButton("Stop", centralwidget)
        self.StopButton.clicked.connect(self.stop_timer)
        self.SaveButton = QtWidgets.QPushButton("Save to EDF", centralwidget)
        self.SaveButton.clicked.connect(self.save_to_edf)

    def setup_layout(self, centralwidget):
        layout = QtWidgets.QVBoxLayout(centralwidget)
        self.data_buffers = {channel: ([], []) for channel in self.channels_to_plot}
        self.setup_figure_and_canvas(layout, centralwidget)
        self.setup_button_layout(layout)
        self.setup_slider_and_listwidget(layout)

        return layout

    def setup_figure_and_canvas(self, layout, MainWindow):
        if self.raw == None: return
        self.signalPlot = SignalPlot(layout, self.channels_to_plot, self.raw)

    def setup_button_layout(self, layout):
        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(self.StopButton)
        buttonLayout.addWidget(self.CloseButton)
        buttonLayout.addWidget(self.SaveButton)
        layout.addLayout(buttonLayout)

    def setup_slider_and_listwidget(self, layout):
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 64)
        self.slider.setValue(0)
        self.slider.sliderReleased.connect(self.slider_update_plot)
        listwidget = self.setup_listwidget()
        slider_Layout = QtWidgets.QHBoxLayout()
        slider_Layout.addWidget(self.slider)
        slider_Layout.addWidget(listwidget)
        layout.addLayout(slider_Layout)

    def setup_listwidget(self):
        listwidget = QListWidget()
        for i, value in enumerate([0, 128, 192, 256, 320, 384, 448, 512, 1024, 2048, 4096]):
            listwidget.insertItem(i, str(value))
        listwidget.clicked.connect(self.clicked)
        return listwidget  

    def epilepsy_prediction(self, data, frequency):
        model_predykcja = tf.keras.models.load_model('./Model/model_new.h5')

        attack = prepare_dataset_attack_model(data, plot_verbose=False)
        a = np.array(attack)
        a = a[np.newaxis, :4]
        y = model_predykcja.predict(a)

        attac_model_save_path = './Model/attack_model_pyTorch.pth'

        loaded_model = AttackModel()
        loaded_model.load_state_dict(torch.load(attac_model_save_path))

        validation_attributes = torch.tensor(y, dtype=torch.float32)
        validation_logits = None

        with torch.no_grad():
            loaded_model.eval()
            validation_logits = loaded_model(validation_attributes).squeeze()
            validation_predictions = torch.round(validation_logits)

        start_sample, end_sample = get_attack_sample_from_predictions(validation_predictions, FRAME_SIZE=1000)
        if start_sample is None or end_sample is None:
            return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        attr_df = data[start_sample:end_sample]

        visualize_predicted_attack(y[0], validation_predictions)

        '''
            Multi channel
        '''

        validation_data = prepare_prediction_multi_channel_datasets(attr_df, plot_verbose=False, rollingN=10)
        channel = 5

        final_predictions = []
        for i in range(len(validation_data)):
            validation_attributes = validation_data[i]["attr"]

            model_save_path = './Model/multi_channel_model_pyTorch 1.pth'

            loaded_model = MultiChannelAttackModel()
            loaded_model.load_state_dict(torch.load(model_save_path))

            validation_attributes = torch.tensor(validation_attributes, dtype=torch.float32)
            validation_logits = None

            with torch.no_grad():
                loaded_model.eval()
                validation_logits = loaded_model(validation_attributes).squeeze()
                validation_predictions = torch.round(validation_logits)

            final_predictions.append(validation_predictions.numpy())
        return np.array(final_predictions).T

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
            self.timer.start()
            self.signalTimer.start()
            self.StopButton.setText(_translate("MainWindow", "Stop"))
        else:
            self.timerStatus = False
            self.timer.stop()
            self.StopButton.setText(_translate("MainWindow", "Start"))

    def load_file(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name = './PN00-4.edf'

        if file_name:
            self.raw = mne.io.read_raw_edf(file_name, preload=True)

    def signal_update_time(self):
        self.source_current_start_idx += self.signalFrequency // 16
        self.source_current_end_idx += self.signalFrequency // 16
        data_length = 0

        for i, channel in enumerate(self.channels_to_plot):
            data, times = self.raw[channel, self.source_current_start_idx:self.source_current_end_idx]

            self.data_buffers[channel][0].extend(data[0])
            self.data_buffers[channel][1].extend(times)
            
            data_length = len(self.data_buffers[channel][0])

        self.data_times += self.signalFrequency // 16

        if data_length > 4096 and (self.data_times % 512 == 0):
            self.data_times = 0
            values_list = np.array(list(self.data_buffers.values()))
            self.temp = self.epilepsy_prediction(values_list[:, 0].T, self.signalFrequency)
            print(self.temp)
            for channel in self.channels_to_plot:
                self.data_buffers[channel] = (
                self.data_buffers[channel][0][-4096:], self.data_buffers[channel][1][-4096:])

        else:
            self.temp = [0] * len(self.channels_to_plot)

    def update_plot(self):
        if self.signalPlot is not None:
            self.signalPlot.update()

    def slider_update_plot(self):
        """Updates the plot based on the slider's current value."""

        start_idx = self.slider.value()
        end_idx = min(self.source_initial_range, start_idx + 64)

        for ax in self.axes:
            ax.cla()

        cmap = plt.get_cmap('tab20')
        colors = cmap(range(len(self.channels_to_plot)))

        for i, channel in enumerate(self.channels_to_plot):
            data, times = self.data_buffers[channel]
            if len(data) > end_idx:
                self.axes[i].plot(times[start_idx:end_idx], data[start_idx:end_idx], label=channel, color=colors[i])
                self.axes[i].legend(loc='upper right')

        self.axes[-1].set_xlabel('Time (s)')

        self.canvas.draw_idle()

    def clicked(self, qmodelindex):
        self.plot_extension = int(self.listwidget.currentItem().text())

    def retranslateUi(self, MainWindow):
        """Retranslates UI elements to the current language."""
        translator = QtCore.QCoreApplication.translate

        MainWindow.setWindowTitle(translator("MainWindow", "MainWindow"))

        self.CloseButton.setText(translator("MainWindow", "Close"))
        self.StopButton.setText(translator("MainWindow", "Start"))

    def save_to_edf(self):
        """Saves the buffered EEG data to an EDF file."""

        channel_names = list(self.data_buffers.keys())
        data_values = np.array(list(self.data_buffers.values()))
        data_values=data_values[:,0].T
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.signalFrequency,
            ch_types='eeg'
        )

        raw = mne.io.RawArray(data_values.T, info)

        file_path = './test.edf'
        mne.export.export_raw(file_path, raw, 'edf', overwrite=True)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())