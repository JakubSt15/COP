import mne


class Normalization:
    min = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def fit(self, three_dementional_tensor):
        for frame in three_dementional_tensor:
            for row_index in range(19):
                min_candidate = min(frame[row_index])
                max_candidate = max(frame[row_index])
                if self.min[row_index] > min_candidate:
                    self.min[row_index] = min_candidate
                if self.max[row_index] < max_candidate:
                    self.max[row_index] = max_candidate

    def transform(self, three_dementional_tensor):
        for frame_index in range(three_dementional_tensor.shape[0]):
            for row_index in range(19):
                a = self.max[row_index] - self.min[row_index]
                for probe_index in range(1000):
                    three_dementional_tensor[frame_index, row_index, probe_index] = (three_dementional_tensor[
                                                                                         frame_index, row_index, probe_index] -
                                                                                     self.min[row_index]) / a
        return three_dementional_tensor

    def transform2d(self, array_2d):
        for row_index in range(19):
            a = self.max[row_index] - self.min[row_index]
            for column_index in range(array_2d.shape[1]):
                array_2d[row_index, column_index] = (array_2d[row_index, column_index] - self.min[row_index]) / a
        return array_2d

    def fit_all(self, list_of_edf):
        for file in list_of_edf:
            edf = mne.io.read_raw_edf(file, preload=True)
            raw_data = mne.make_fixed_length_epochs(edf, 2)
            raw_data = raw_data.get_data()
            self.fit(raw_data)
