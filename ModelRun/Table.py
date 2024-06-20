from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QFont, QColor



class GuiAttackTable():
    def __init__(self, layout, channels_to_plot):
        self.channels = channels_to_plot
        self.table = QTableWidget(2, len(channels_to_plot))
        self.table.setFont(QFont("Arial", 10))
        column_width = 4
        for col, channel in enumerate(channels_to_plot):
            channel_item = QTableWidgetItem(channel[-2:])
            self.table.setColumnWidth(col, column_width)
            self.table.setItem(0, col, channel_item) 

        self.table.setFixedSize(1120, 100)
        layout.addWidget(self.table)


    '''
        values should be an array of dicts
        [
            {
                channelName: string
                attackProbability: float <0.0, 1.0>
            }
        ]
    '''
    def updateTable(self, values):
        self.table.clearContents()
        for i, channel_info in enumerate(values):
            col = self.channels.index(self.channels[i])

            color = self._getColorBasedOnPrediction(channel_info, i)
            probability_item = QTableWidgetItem(f"{100*channel_info:.0f}%")
            probability_item.setBackground(color)
            self.table.setItem(1, col, probability_item)
            channel_item = QTableWidgetItem(self.channels[i][-2:])
            channel_item.setBackground(color)
            self.table.setItem(0, col, channel_item)


    def _getColorBasedOnPrediction(self, prediction, id):
        if prediction > 0.9:
            return QColor('#fc0303')  # Red
        elif prediction > 0.8:
            return QColor('#fc3503')  # Orange
        elif prediction > 0.7:
            return QColor('#fc6b03')  # Dark Orange
        elif prediction > 0.6:
            return QColor('#fcb503')  # Yellow
        elif prediction > 0.5:
            return QColor('#fcf403')  # Light Yellow
        elif prediction > 0.25:
            return QColor('#bafc03')  # Light Green
        else:
            return [QColor('#31f766'), QColor('#008a25')][id % 2]