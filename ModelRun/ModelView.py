import json
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

tf.get_logger().setLevel('FATAL')


class Ui_MainWindow(object):
    def __init__(self, MainWindow, file_name=None):
        self.table = None
        self.signalPlot = None
        self.predictionTables = None
        self.raw = None
        self.threesholds = None
        self.setup_initial_values()
        self.setupUi(MainWindow)

        ''' Timer for updating plots '''
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)  # Update interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.stop()

        self.setup_initial_values()  # Teraz inicjalizacja jest pierwsza

        if file_name is not None:
            self.load_file(file_name)

        self.isPlotting = False
        self.setupUi(MainWindow)

        ''' CSV save data (exact time of start, end record, prediction buffer )'''
        self.timeInitialized = False
        self.startTime = None
        self.endTime = None
        self.predictionsBuffer = []

    def setup_initial_values(self):
        mne.set_log_level('CRITICAL')
        self.isPlotting = False
        self.followPlot = False
        self.signalFrequency = 512
        self.elapsed_time = 0
        self.start_time = 0
        self.plot_extension = 0
        self.data_times = 32
        self.threesholds= {
            'High Risk': 90,
            'Medium High Risk': 80,
            'Medium Risk': 70,
            'Medium Low Risk': 60,
            'Low Risk': 50,
            'Very Low Risk': 25,
            "Warning enabled": False
         }
        self.readThreesholds()
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

    def setupUi(self, MainWindow):
        self.setup_main_window(MainWindow)
        self.centralwidget = self.setup_central_widget(MainWindow)
        self.setup_buttons(self.centralwidget)  # Najpierw tworzymy przyciski
        layout = self.setup_layout(self.centralwidget)  # Potem dodajemy je do layoutu
        self.retranslateUi(MainWindow)
        self.load_file()

        self.retranslateUi(MainWindow)
        self.apply_styles()

        if self.raw is not None and self.isPlotting:
            self.timer.start()

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

        self.ChooseButton = QtWidgets.QPushButton("Pick EDF", centralwidget)
        self.ChooseButton.clicked.connect(self.change_edf)

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
        self.SaveButton.clicked.connect(self.signalPlot.save_to_edf)
        layout.insertWidget(0, self.signalPlot.plotWidget)
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
        buttonLayout.addWidget(self.ChooseButton)
        buttonLayout.addWidget(self.SaveCSVButton)
        layout.addWidget(buttonBox)

    def setup_table(self, layout):
        self.table = GuiAttackTable(layout, self.channels_to_plot)

    def setup_color_legend(self, layout):
        legend_layout = QtWidgets.QHBoxLayout()

        # Define the legend colors and corresponding labels
        legend_data = [
            (self.threesholds['high'], '#fc0303', 'High Risk >'+str(self.threesholds['high'])+'%'),
            (self.threesholds['medium_high'], '#fc3503', 'Medium High Risk >'+str(self.threesholds['medium_high'])+'%'),
            (self.threesholds['medium'], '#fc6b03', 'Medium Risk >'+str(self.threesholds['medium'])+'%'),
            (self.threesholds['medium_low'], '#fcb503', 'Medium Low Risk >'+str(self.threesholds['medium_low'])+'%'),
            (self.threesholds['low'], '#fcf403', 'Low Risk >'+str(self.threesholds['low'])+'%'),
            (self.threesholds['very_low'], '#bafc03', 'Very Low Risk >'+str(self.threesholds['very_low'])+'%')
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
            ret = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        self.ChooseButton.setStyleSheet(button_style)
        self.SaveCSVButton.setStyleSheet(button_style)

    def stop_plot(self):
        _translate = QtCore.QCoreApplication.translate
        if not self.isPlotting:
            self.isPlotting = True
            self.followPlot = True
            self.timer.start()
            self.setStartRecordTime()
            self.StopButton.setText(_translate("MainWindow", "Stop"))
        else:
            self.isPlotting = False
            self.followPlot = False
            self.StopButton.setText(_translate("MainWindow", "Start"))

    def load_file(self, file_name='./PN00-4.edf'):
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

    def reset_plot(self):
        """Clears the old plot and resets all relevant variables."""

        # Clear and remove main signal plot
        if self.signalPlot is not None:
            self.signalPlot.clear_plot(self.data_times)  # Pass data_times argument
            self.signalPlot.plotWidget.deleteLater()
            self.signalPlot = None

        # Clear prediction plot and buffers
        if self.predictionTables is not None:
            self.predictionTables.plotWidget.clear()
            self.predictionTables.predictionsBuffer.clear()
            self.predictionTables.realDataBuffer.clear()
            self.predictionTables.init_empty_plot(self.predictionTables.plotWidget, title="Prediction Plot")

        # Reset other variables
        self.timer.stop()
        self.data_buffers = {channel: ([], []) for channel in self.channels_to_plot}
        self.elapsed_time = 0
        self.start_time = 0
        self.plot_extension = 0
        self.source_current_start_idx = 0
        self.source_current_end_idx = 32

        # Re-initialize signalPlot
        signalPlotHeight = 620  # Or get it from the layout
        layout = self.centralwidget.layout()
        self.setup_figure_and_canvas(layout, signalPlotHeight)

    def change_edf(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open EDF File", "", "EDF Files (*.edf);;All Files (*)", options=options
        )
        if file_name:
            self.load_file(file_name)
            self.reset_plot()
            self.StopButton.setText("Start")
            self.isPlotting = False

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

    def readThreesholds(self):
        with open("thresholds.json", "r") as file:
            data = json.load(file)
            self.threesholds = data


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
