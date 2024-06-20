from PyQt5 import QtWidgets, QtCore
import mne
import numpy as np
import pyqtgraph as pg

from DoctorMenu import DoctorMenuList

class SignalPlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.channels_to_plot = ['eeg fp1', 'eeg f3', 'eeg c3',
                                 'eeg p3', 'eeg o1', 'eeg f7', 'eeg t3',
                                 'eeg t5', 'eeg fz', 'eeg cz', 'eeg pz',
                                 'eeg fp2', 'eeg f4', 'eeg c4', 'eeg p4',
                                 'eeg o2', 'eeg f8', 'eeg t4', 'eeg t6']

        self.plotWidget = pg.PlotWidget()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.plotWidget)

        self.maxLen = 5

        self.initPlotHandler()

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.sliderReleased.connect(self.update_plot)

        # self.layout.addWidget(self.slider)

        self.current_start_idx = 0
        self.current_end_idx = None
        self.channel_map = {}
        self.raw = None

    def initPlotHandler(self):
        self.plotHandlers = []
        self.plotWidget.setXRange(0, self.maxLen, padding=0.1)
        self.plotWidget.setYRange(0.001, 0.001)
        self.plotWidget.showGrid(x=True, y=True)
        self.plotWidget.setLabel('left', 'Amplitude')
        self.plotWidget.setLabel('bottom', 'Time')
        self.plotWidget.setMouseEnabled(y=True)


        self.offset = 0.00004  # Offset between channels

        for i, channel in enumerate(self.channels_to_plot):
            plot_item = self.plotWidget.plotItem
            plot_item.setDownsampling(auto=True, mode='subsample')
            plot = plot_item.plot(pen=pg.mkPen(color=(i, len(self.channels_to_plot)*1.3)))
            self.plotHandlers.append(plot)



    def load_data(self, file_name):
        self.raw = mne.io.read_raw_edf(file_name, preload=True)
        self.n_samples = len(self.raw.times)
        self.current_end_idx = self.n_samples
        self.slider.setRange(0, self.n_samples)

        # Create a mapping from normalized channel names to original names
        self.channel_map = {ch.lower(): ch for ch in self.raw.ch_names}

        self.update_plot()

    def update_plot(self):
        slider_value = self.slider.value()
        self.current_start_idx = 0
        self.current_end_idx = self.n_samples

        for i, channel in enumerate(self.channels_to_plot):
            original_channel = self.channel_map.get(channel)
            if original_channel is not None:
                data, times = self.raw[original_channel, self.current_start_idx:self.current_end_idx]
                self.plotHandlers[i].setData(times, data[0] + (i * self.offset))

        # Auto scale Y axis to fit the data range
        self.plotWidget.enableAutoRange(axis=pg.ViewBox.YAxis)

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("EDF View")
        MainWindow.resize(946, 578)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 946, 26))
        self.menubar.setObjectName("menubar")
        self.menuStrona_Glowna = QtWidgets.QMenu(self.menubar)
        self.menuStrona_Glowna.setObjectName("menuStrona_Glowna")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionLoad_File = QtWidgets.QAction(MainWindow)
        self.actionLoad_File.setObjectName("actionLoad_File")
        self.menuStrona_Glowna.addAction(self.actionLoad_File)
        self.menubar.addAction(self.menuStrona_Glowna.menuAction())
        self.actionLoad_File.triggered.connect(self.load_file)

        self.CloseButton = QtWidgets.QPushButton("Close", self.centralwidget)
        self.CloseButton.clicked.connect(self.close_window)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.signal_plot = SignalPlot()
        self.layout.addWidget(self.signal_plot)
        self.layout.addWidget(self.CloseButton)

        self.channel_map = {}

    def close_window(self):
        QtWidgets.qApp.closeAllWindows()
        self.addUser = QtWidgets.QMainWindow()
        self.secondWindow = DoctorMenuList.Ui_MainWindow()
        self.secondWindow.setupUi(self.addUser)
        self.addUser.show()

    def load_file(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", "", "EDF Files (*.edf);;All Files (*)",
                                                  options=options)

        if file_name:
            self.signal_plot.load_data(file_name)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("EDF View", "EDF View"))
        self.menuStrona_Glowna.setTitle(_translate("EDF View", "File"))
        self.actionLoad_File.setText(_translate("EDF View", "Load File"))


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
