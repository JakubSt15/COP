import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy, QWidget

class SignalPlot(QWidget):
    def __init__(self, layout, channels, data, parent=None):
        super(SignalPlot, self).__init__(parent)
        self.layout = layout
        self.channels = channels
        self.signalPlotData = self.filter_channels(data, self.channels)

        self.figure, self.axes = plt.subplots(len(channels), 1, sharex=True, figsize=(15, 10))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.layout.addWidget(self.canvas, 2, 0, 1, 2)
        self.update()

    def filter_channels(self, data, channels):
        channels_lower = {ch.lower(): ch for ch in channels}
        return {channels_lower[ch.lower()]: data[ch] for ch in data if ch.lower() in channels_lower}

    def update(self):
        self.figure.clear()

        for i, ch in enumerate(self.channels):
            ch_data = self.signalPlotData[ch][0][0]  # Extract channel data
            self.axes[i].cla()  # Clear previous plot
            self.axes[i].plot(ch_data, color='#31f766')  # Plot data with specified color
            self.axes[i].set_title(ch)
            self.axes[i].grid(True)

        self.figure.tight_layout()
        self.canvas.draw()
