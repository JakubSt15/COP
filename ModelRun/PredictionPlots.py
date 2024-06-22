import pyqtgraph as pg
from collections import deque


class PredictionPlots():
    def __init__(self, layout):
        self.plotWidget = pg.PlotWidget()

        self.plotWidget.setFixedHeight(210)

        self.init_empty_plot(self.plotWidget, title="Prediction Plot")

        layout.addWidget(self.plotWidget)

        self.predictionsBuffer = deque()
        self.realDataBuffer = deque()

    def init_empty_plot(self, plot_widget, title=""):
        plot = plot_widget.getPlotItem()
        plot.setTitle(title)
        plot.showGrid(x=True, y=True)  # Show grid lines
        plot.setLabel('left', 'Value')
        plot.setLabel('bottom', 'Time')
        plot.addLegend()
        plot.setMouseEnabled(x=False, y=False)
        plot_widget.plot()


    def upgradePredictionPlot(self):
        x = [0.0, 0.25, 0.5, 1.0]
        self.plotWidget.clear() 
        self.plotWidget.plot(x, self.predictionsBuffer[0], pen=pg.mkPen('r', width=2), symbol='o', symbolSize=10, symbolBrush=('r'), name='predicted signal power')  # Plot with red lines and points
        
        if len(self.realDataBuffer) != 6: return
        self.plotWidget.plot(x, self.realDataBuffer[0], pen=pg.mkPen('b', width=2), symbol='o', symbolSize=10, symbolBrush=('b'), name='actual signal power')  # Plot with red lines and points

    def pushNewPrediction(self, values):
        self.predictionsBuffer.append(values)
        if len(self.predictionsBuffer) > 6:
            self.predictionsBuffer.popleft()

    def pushNewRealData(self, values):
        self.realDataBuffer.appendleft(values)
        if len(self.realDataBuffer) > 6:
            self.realDataBuffer.pop()