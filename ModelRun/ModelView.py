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
from CommonTools.CommonTools import show_popup
from PyQt5.QtWidgets import QFileDialog
from DoctorMenu import DoctorMenuList
from Model.prepare_data import prepare_dataset_attack_model, get_attack_sample_from_predictions, \
    prepare_prediction_multi_channel_datasets
from Model.train_attack import AttackModel, MultiChannelAttackModel
from Model.visualize import visualize_predicted_attack
from ModelRun.Graph import SignalPlot

import pyqtgraph as pg
import logging

plt.style.use('dark_background')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('FATAL')


class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        self.signalPlot = None
        self.raw = None
        self.setup_initial_values()
        self.setupUi(MainWindow)

        ''' Timer for updating plots '''
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)  # Update interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.stop()

    def setup_initial_values(self):
        mne.set_log_level('CRITICAL')
        self.isPlotting = False
        self.signalFrequency = 512
        self.elapsed_time = 0
        self.start_time = 0
        self.plot_extension = 0
        self.data_times = 32
        self.channels_to_plot = ['eeg fp1', 'eeg f3', 'eeg c3',
                                 'eeg p3', 'eeg o1', 'eeg f7', 'eeg t3',
                                 'eeg t5', 'eeg fz', 'eeg cz', 'eeg pz',
                                 'eeg fp2', 'eeg f4', 'eeg c4', 'eeg p4',
                                 'eeg o2', 'eeg f8', 'eeg t4', 'eeg t6']
        self.temp = [0] * len(self.channels_to_plot)
        self.source_initial_range = 100
        self.source_current_start_idx = 0
        self.source_current_end_idx = 32

    def setupUi(self, MainWindow):
        self.setup_main_window(MainWindow)
        centralwidget = self.setup_central_widget(MainWindow)
        self.setup_buttons(centralwidget)
        layout = self.setup_layout(centralwidget)
        self.load_file()
        self.retranslateUi(MainWindow)
        self.apply_styles()
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
        self.StopButton.clicked.connect(self.stop_plot)
        self.SaveButton = QtWidgets.QPushButton("Save to EDF", centralwidget)
        self.SaveButton.clicked.connect(self.save_to_edf)

    def setup_layout(self, centralwidget):
        layout = QtWidgets.QVBoxLayout(centralwidget)
        self.data_buffers = {channel: ([], []) for channel in self.channels_to_plot}
        self.setup_figure_and_canvas(layout, centralwidget)
        self.setup_button_layout(layout)

        return layout

    def setup_figure_and_canvas(self, layout, MainWindow):
        if self.raw == None: return
        self.signalPlot = SignalPlot(layout, self.channels_to_plot, self.raw, self.epilepsy_prediction)

    def setup_button_layout(self, layout):
        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(self.StopButton)
        buttonLayout.addWidget(self.CloseButton)
        buttonLayout.addWidget(self.SaveButton)
        layout.addLayout(buttonLayout)

    def epilepsy_prediction(self, data, frequency, predictProba=False):
        model_predykcja = tf.keras.models.load_model('./Model/model.keras')
        attack = prepare_dataset_attack_model(data, plot_verbose=False)
        a = np.array(attack)
        a = a[np.newaxis, :4]
        logging.getLogger("absl").setLevel(logging.ERROR)
        y = model_predykcja.predict(a, verbose=0)

        attac_model_save_path = './Model/attack_model_pyTorch.pth'

        loaded_model = AttackModel()
        loaded_model.load_state_dict(torch.load(attac_model_save_path))

        validation_attributes = torch.tensor(y, dtype=torch.float32)
        validation_logits = None

        with torch.no_grad():
            loaded_model.eval()
            validation_logits = loaded_model(validation_attributes).squeeze()
            validation_predictions = validation_logits
            if not predictProba: validation_predictions = torch.round(validation_logits)

        print("WILL BE ATTACK: ", validation_logits)
        start_sample, end_sample = get_attack_sample_from_predictions(validation_predictions, FRAME_SIZE=100)
        
        if start_sample is None or end_sample is None:
            return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        attr_df = data[start_sample:end_sample]

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
                validation_predictions = validation_logits
                if not predictProba: validation_predictions = torch.round(validation_logits)

            final_predictions.append(validation_predictions.numpy())
            
        return np.array(final_predictions).T

    def close_window(self):
        QtWidgets.qApp.closeAllWindows()
        self.doctorMenu = QtWidgets.QMainWindow()
        self.secondWindow = DoctorMenuList.Ui_MainWindow()
        self.secondWindow.setupUi(self.doctorMenu)
        self.doctorMenu.show()

    def apply_styles(self):
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
                border: 2px solid #4CAF50;
            }
        """
        button_style_reversed = """
            QPushButton {
                background-color: #AF4CAB;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
                border: 2px solid #AF4CAB;
            }
        """
        self.CloseButton.setStyleSheet(button_style_reversed)
        self.StopButton.setStyleSheet(button_style)
        self.SaveButton.setStyleSheet(button_style)

    def stop_plot(self):
        _translate = QtCore.QCoreApplication.translate
        if self.isPlotting == False:
            self.StopButton.setText(_translate("MainWindow", "Stop"))
            self.isPlotting = True
            self.timer.start()
        else:
            self.isPlotting = False
            self.timer.stop()
            self.StopButton.setText(_translate("MainWindow", "Start"))

    def load_file(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name = './PN00-4.edf'

        if file_name:
            self.raw = mne.io.read_raw_edf(file_name, preload=True)

    def update_plot(self):
        if self.signalPlot is not None:
            self.signalPlot.update()

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

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(None, "Wybierz nazwę pliku", "", "EDF Files (*.edf)", options=options)
        if file_path:
            if not file_path.endswith('.edf'):
                file_path += '.edf'
            mne.export.export_raw(file_path, raw, 'edf', overwrite=True)
            show_popup("Zapisano", f"Plik EDF zapisano w: {file_path}")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
