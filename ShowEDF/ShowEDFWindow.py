# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from datetime import timedelta

from DoctorMenu import DoctorMenuList

plt.style.use('dark_background')


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("EDF View")
        MainWindow.resize(946, 578)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 946, 26))
        self.menubar.setObjectName("menubar")
        self.menuStrona_G_wna = QtWidgets.QMenu(self.menubar)
        self.menuStrona_G_wna.setObjectName("menuStrona_G_wna")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionLoad_File = QtWidgets.QAction(MainWindow)
        self.actionLoad_File.setObjectName("actionLoad_File")
        self.menuStrona_G_wna.addAction(self.actionLoad_File)
        self.menubar.addAction(self.menuStrona_G_wna.menuAction())
        self.actionLoad_File.triggered.connect(self.load_file)

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

        self.figure, self.axes = plt.subplots(len(self.channels_to_plot), 1, sharex=True, figsize=(10, 20))
        plt.subplots_adjust(bottom=0.05, left=0.005, top=0.95)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, MainWindow)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.sliderReleased.connect(self.update_plot)

        self.layout = QtWidgets.QGridLayout(self.centralwidget)
        self.layout.addWidget(self.toolbar, 0, 0, 1, 2)
        self.layout.addWidget(self.canvas, 1, 0, 1, 2)
        self.layout.addWidget(self.slider, 2, 0, 1, 2)
        self.layout.addWidget(self.CloseButton, 3, 0, 1, 2)

        self.layout.setRowStretch(1, 1)
        self.layout.setRowStretch(2, 1)

        self.initial_range = 20000
        self.current_start_idx = 0
        self.channel_map = {}

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
            self.n_samples = len(self.raw.times)
            self.current_end_idx = min(self.n_samples, self.initial_range)
            self.slider.setRange(0, self.n_samples - self.initial_range)

            # Create a mapping from normalized channel names to original names
            self.channel_map = {ch.lower(): ch for ch in self.raw.ch_names}
            self.update_plot()

    def update_plot(self):
        slider_value = self.slider.value()
        self.current_start_idx = slider_value
        self.current_end_idx = min(self.n_samples, self.current_start_idx + self.initial_range)

        for ax in self.axes:
            ax.clear()

        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(len(self.channels_to_plot))]

        for i, channel in enumerate(self.channels_to_plot):
            original_channel = self.channel_map.get(channel)
            if original_channel is not None:
                data, times = self.raw[original_channel, self.current_start_idx:self.current_end_idx]
                self.axes[i].plot(times, data[0], label=original_channel, color=colors[i])
                self.axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                self.axes[i].set_yticklabels([])
                self.axes[i].spines['bottom'].set_visible(False)
                self.axes[i].spines['right'].set_visible(False)
                self.axes[i].spines['left'].set_visible(False)

        self.axes[-1].set_xlabel('Time')
        self.axes[-1].set_xticklabels(
            [str(timedelta(seconds=int(sec))) for sec in self.raw.times[self.current_start_idx:self.current_end_idx]])
        self.canvas.draw()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("EDF View", "EDF View"))
        self.menuStrona_G_wna.setTitle(_translate("EDF View", "File"))
        self.actionLoad_File.setText(_translate("EDF View", "Load File"))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ShowEDF = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(ShowEDF)
    ShowEDF.show()
    sys.exit(app.exec())
