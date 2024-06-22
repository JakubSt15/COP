import os
import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QListWidget, QTableWidget, QTableWidgetItem, QWidget, QVBoxLayout, QApplication
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
import matplotlib.pyplot as plt
import mne
from CommonTools.CommonTools import show_popup
from PyQt5.QtWidgets import QFileDialog
from DoctorMenu import DoctorMenuList
from Model.prepare_data import prepare_dataset_attack_model, get_attack_sample_from_predictions, \
    prepare_prediction_multi_channel_datasets
from Model.train_attack import AttackModel, MultiChannelAttackModel
from ModelRun.Graph import SignalPlot
from ModelRun.Table import GuiAttackTable
from ModelRun.PredictionPlots import PredictionPlots
import csv
import logging
from datetime import datetime

plt.style.use('dark_background')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('FATAL')


class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        self.table = None
        self.signalPlot = None
        self.predictionTables = None
        self.raw = None
        self.setup_initial_values()
        self.setupUi(MainWindow)

        ''' Timer for updating plots '''
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)  # Update interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.stop()

        ''' CSV save data (exact time of start, end record, prediction buffer )'''
        self.timeInitialized = False
        self.startTime = None
        self.endTime = None
        self.predictionsBuffer = []

    def setup_initial_values(self):
        mne.set_log_level('CRITICAL')
        self.followPlot = False
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
        self.SaveCSVButton = QtWidgets.QPushButton("Save attack info CSV", centralwidget)
        self.SaveCSVButton.clicked.connect(self.save_csv)

    def setup_layout(self, centralwidget):
        layout = QtWidgets.QVBoxLayout(centralwidget)

        ''' Fixed values of layout - in px '''
        width = 1720        
        height = 920 
        signalPlotHeight = 620
        buttonSize = (160, 20)
        centralwidget.setFixedHeight(height)
    
        self.data_buffers = {channel: ([], []) for channel in self.channels_to_plot}
        self.setup_figure_and_canvas(layout, signalPlotHeight)

        bottomContainer = QtWidgets.QHBoxLayout()
        bottomRightContainer = QtWidgets.QVBoxLayout()
        bottomLeftContainer = QtWidgets.QVBoxLayout()


        self.setup_button_layout(bottomLeftContainer, buttonSize)
        self.setup_table(bottomRightContainer)
        self.setup_color_legend(bottomRightContainer)
        self.setup_prediction_plots(bottomLeftContainer)

        bottomContainer.addLayout(bottomLeftContainer)
        spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        bottomContainer.addItem(spacer)
        bottomContainer.addLayout(bottomRightContainer)
        layout.addLayout(bottomContainer)
        return layout

    def setup_figure_and_canvas(self, layout, signalPlotHeight):
        if self.raw == None: return
        self.signalPlot = SignalPlot(layout, self.channels_to_plot, self.raw, self.epilepsy_prediction)
        self.signalPlot.plotWidget.setFixedHeight(signalPlotHeight)

    def setup_button_layout(self, layout, buttonSize):
        buttonBox = QtWidgets.QGroupBox()
        buttonLayout = QtWidgets.QHBoxLayout(buttonBox)
        # self.StopButton.setFixedSize(buttonSize[0], buttonSize[1])
        # self.CloseButton.setFixedSize(buttonSize[0], buttonSize[1])
        # self.SaveButton.setFixedSize(buttonSize[0], buttonSize[1])
        # self.SaveCSVButton.setFixedSize(buttonSize[0], buttonSize[1])
        buttonLayout.addWidget(self.StopButton)
        buttonLayout.addWidget(self.CloseButton)
        buttonLayout.addWidget(self.SaveButton)
        buttonLayout.addWidget(self.SaveCSVButton)
        layout.addWidget(buttonBox)

    def setup_table(self, layout):
        self.table = GuiAttackTable(layout, self.channels_to_plot)

    def setup_color_legend(self, layout):
        legend_layout = QtWidgets.QHBoxLayout()

        # Define the legend colors and corresponding labels
        legend_data = [
            (0.9, '#fc0303', 'High Risk >90%'),
            (0.8, '#fc3503', 'Medium High Risk >80%'),
            (0.7, '#fc6b03', 'Medium Risk >70%'),
            (0.6, '#fcb503', 'Medium Low Risk >60%'),
            (0.5, '#fcf403', 'Low Risk >50%'),
            (0.25, '#bafc03', 'Very Low Risk >25%')
        ]

        for threshold, color, label_text in legend_data:
            color_widget = QtWidgets.QWidget()
            color_widget.setFixedSize(20, 20)  # Size of the colored rectangle
            color_widget.setStyleSheet(f'background-color: {color}; border: 1px solid black;')

            label = QtWidgets.QLabel(label_text)
            label.setStyleSheet('font-weight: bold;')

            # Create a vertical layout for each legend item
            item_layout = QtWidgets.QVBoxLayout()
            item_layout.addWidget(color_widget, alignment=QtCore.Qt.AlignCenter)
            item_layout.addWidget(label, alignment=QtCore.Qt.AlignCenter)

            legend_layout.addLayout(item_layout)

        layout.addLayout(legend_layout)

    def setup_prediction_plots(self, layout):
        doublePlotContainer = QtWidgets.QHBoxLayout()
        self.predictionTables = PredictionPlots(doublePlotContainer)
        layout.addLayout(doublePlotContainer)

    def epilepsy_prediction(self, data, frequency, predictProba=False):
        model_predykcja = tf.keras.models.load_model('./Model/model.keras')
        attack = prepare_dataset_attack_model(data, plot_verbose=False)
        a = np.array(attack)
        a = a[np.newaxis, :4]
        logging.getLogger("absl").setLevel(logging.ERROR)
        y = model_predykcja.predict(a, verbose=0)
        
        ''' append to Gui prediction plots buffers'''
        channelToDisplay = 0
        self.predictionTables.pushNewRealData(a[:, :, channelToDisplay].flatten())
        self.predictionTables.pushNewPrediction(y[:, :, channelToDisplay].flatten())
        
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

        start_sample, end_sample = get_attack_sample_from_predictions(validation_predictions, FRAME_SIZE=100)
        
        if start_sample is None or end_sample is None:
            ret = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            self.predictionsBuffer.append(ret)
            return ret
        
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
                padding: 6px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 12px;
                border-radius: 4px;
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
                padding: 6px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 12px;
                border-radius: 4px;
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
        self.SaveCSVButton.setStyleSheet(button_style)

    def stop_plot(self):
        _translate = QtCore.QCoreApplication.translate
        if self.followPlot == False:
            self.StopButton.setText(_translate("MainWindow", "Stop"))
            self.followPlot = True
            self.timer.start()
            self.setStartRecordTime()
        else:
            self.followPlot = False
            self.StopButton.setText(_translate("MainWindow", "Start"))

    def load_file(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name = './PN00-4.edf'

        if file_name:
            self.raw = mne.io.read_raw_edf(file_name, preload=True)

    def update_plot(self):
        updated_attack_proba = self.signalPlot.update(followPlot=self.followPlot)
        if updated_attack_proba is not None:
            self.table.updateTable(updated_attack_proba)
            self.predictionsBuffer.append(updated_attack_proba)
            self.predictionTables.upgradePredictionPlot()

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

    def save_csv(self):
        if not self.timeInitialized: return
        self.setEndRecordTime()
        columns = self.channels_to_plot
        filename = f'SavedRecords/{self.startTime.replace(":", "_")}--{self.endTime.replace(":", "_")}.csv'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(self.predictionsBuffer)
        show_popup("Saved", f"Report saved in : {filename}", QtWidgets.QMessageBox.Information)

    def setStartRecordTime(self):
        if self.timeInitialized == True: return
        self.timeInitialized = True
        self.startTime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    def setEndRecordTime(self):
        if self.timeInitialized == False: return
        self.endTime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.timeInitialized = False

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
