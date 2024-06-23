# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from datetime import timedelta
from ShowEDF.Graph import SignalPlot
from DoctorMenu import DoctorMenuList

plt.style.use('dark_background')


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        mne.set_log_level('CRITICAL')
        MainWindow.setObjectName("EDF View")
        MainWindow.resize(946, 578)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        # Create Button Load File
        self.LoadFileButton = QtWidgets.QPushButton("Load File", self.centralwidget)
        self.LoadFileButton.clicked.connect(self.load_file)

        # Create Button Close All plots
        self.ClosePlotsButton = QtWidgets.QPushButton("Close All Plots", self.centralwidget)
        self.ClosePlotsButton.clicked.connect(self.close_plots)

        # Create buttons without setting geometry
        self.CloseButton = QtWidgets.QPushButton("Close", self.centralwidget)
        self.CloseButton.clicked.connect(self.close_window)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.channels_to_plot = ['eeg fp1', 'eeg f3', 'eeg c3',
                                 'eeg p3', 'eeg o1', 'eeg f7', 'eeg t3',
                                 'eeg t5', 'eeg fz', 'eeg cz', 'eeg pz',
                                 'eeg fp2', 'eeg f4', 'eeg c4', 'eeg p4',
                                 'eeg o2', 'eeg f8', 'eeg t4', 'eeg t6']

        self.layout = QtWidgets.QGridLayout(self.centralwidget)
        self.layout.addWidget(self.LoadFileButton, 2, 0, 1, 1)
        self.layout.addWidget(self.ClosePlotsButton, 3, 0, 1, 1)
        self.layout.addWidget(self.CloseButton, 4, 0, 1, 1)

        self.channel_map = {}
        self.setStyles()

    def close_plots(self):
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget and isinstance(widget, QtWidgets.QPushButton):
                continue  # Nie usuwaj przycisków
            else:
                self.layout.itemAt(i).widget().setParent(None)

    def close_window(self):
        QtWidgets.qApp.closeAllWindows()
        self.doctorMenu = QtWidgets.QMainWindow()
        self.secondWindow = DoctorMenuList.Ui_MainWindow()
        self.secondWindow.setupUi(self.doctorMenu)
        self.doctorMenu.show()

    def load_file(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", "", "EDF Files (*.edf);;All Files (*)",
                                                             options=options)

        if file_name:
            self.raw = mne.io.read_raw_edf(file_name, preload=True)
            self.show_plot()

    def show_plot(self):
        self.signalPlot = SignalPlot(self.layout, self.raw, self.channels_to_plot)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("EDF View", "EDF View"))

    def setStyles(self):
        # Set button style
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
        self.CloseButton.setStyleSheet(button_style_reversed)
        self.LoadFileButton.setStyleSheet(button_style)
        self.ClosePlotsButton.setStyleSheet(button_style)



if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ShowEDF = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(ShowEDF)
    ShowEDF.show()
    sys.exit(app.exec())
