import pyqtgraph as pg



class PredictionPlots():
    def __init__(self, layout):
        self.predictionPlotWidget = pg.PlotWidget()
        self.realPlotWidget = pg.PlotWidget()

        self.predictionPlotWidget.setFixedHeight(210)
        self.realPlotWidget.setFixedHeight(210)

        self.init_empty_plot(self.predictionPlotWidget, title="Prediction Plot")
        self.init_empty_plot(self.realPlotWidget, title="Real Plot")

        layout.addWidget(self.predictionPlotWidget)
        layout.addWidget(self.realPlotWidget)

    def init_empty_plot(self, plot_widget, title=""):
        plot = plot_widget.getPlotItem()
        plot.setTitle(title)
        plot.showGrid(x=True, y=True)  # Show grid lines
        plot.setLabel('left', 'Value')
        plot.setLabel('bottom', 'Time')
        plot.setMouseEnabled(x=False, y=False)
        plot_widget.plot()


    def upgradePredictionPlot(self, values):
        x = [0.0, 0.25, 0.5, 1.0]
        self.predictionPlotWidget.clear() 
        self.predictionPlotWidget.plot(x, values, pen=pg.mkPen('r', width=2), symbol='o', symbolSize=10, symbolBrush=('r'))  # Plot with red lines and points

    def upgradeRealPlot(self, values):
        x = [0.0, 0.25, 0.5, 1.0]
        self.predictionPlotWidget.clear() 
        self.predictionPlotWidget.plot(x, values, pen=pg.mkPen('r', width=2), symbol='o', symbolSize=10, symbolBrush=('r'))  # Plot with red lines and points