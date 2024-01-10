from scipy import signal
class Filter:
    def __init__(self):
        self.low_cut=3
        self.high_cut=25
    def butter_filter(self, raw_data, frequency):
        b, a = signal.butter(N=6, Wn=[self.low_cut, self.high_cut], btype='bandpass', fs=frequency, analog=False)
        for channel in range(19):
            raw_data.iloc[channel] = signal.lfilter(b, a, raw_data.iloc[channel])
        return raw_data