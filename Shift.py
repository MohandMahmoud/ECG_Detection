import os
import numpy as np


class TemplateMatcher:
    @staticmethod
    def load_signals_from_folder(folder_path):
        signals = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                signal = np.loadtxt(file_path)
                signals.append(signal)
        return signals

    @staticmethod
    def coor(Y_3, Y_4):

        sum_of_squares = sum([q ** 2 for q in Y_3])
        out = []

        def shift_left(lst):
            first_element = lst[0]
            shifted_lst = lst[1:] + [first_element]
            return shifted_lst

        original_list = Y_4
        e = 0
        for i in range(len(original_list)):
            for o in range(len(original_list)):
                e += Y_3[o] * original_list[o]
            e = (1 / 5) * e
            sum_of_squares_Y = sum([n ** 2 for n in original_list])
            b = sum_of_squares * sum_of_squares_Y
            v = np.sqrt(b)
            t = (1 / 5) * v
            p = e / t
            out.append(p)
            shifted_list = shift_left(original_list)
            original_list = shifted_list
            e = 0
        print(out)
        return out

    def template_matching(self, test_signal, templates):
        correlations = self.coor(test_signal, templates)
        predicted_class = np.argmax(correlations)
        return predicted_class - 1

    @staticmethod
    def compute_templates(class_signals):
        return [np.mean(signal, axis=0) for signal in class_signals]

    def matching(self):
        class1_folder = "C:/Users/Lenovo/PycharmProjects/Honda/Class 1-20231202T200043Z-001/Class 1"
        class2_folder = "C:/Users/Lenovo/PycharmProjects/Honda/Class 2-20231202T200044Z-001/Class 2"
        test_folder = "C:/Users/Lenovo/PycharmProjects/Honda/Test Signals-20231202T200046Z-001/Test Signals"

        class1_signals = self.load_signals_from_folder(class1_folder)
        class2_signals = self.load_signals_from_folder(class2_folder)
        test_signals = self.load_signals_from_folder(test_folder)

        class1_templates = self.compute_templates(class1_signals)
        class2_templates = self.compute_templates(class2_signals)

        predictions = []
        for test_signal in test_signals:
            predicted_class = self.template_matching(test_signal, class1_templates + class2_templates)
            predictions.append(predicted_class)

        for i, predicted_class in enumerate(predictions):
            if predicted_class == 1:
                print("Test Signal ", i + 1, " Predicted Class is down")
            else:
                print("Test Signal ", i + 1, " Predicted Class is up")


template_matcher = TemplateMatcher()
template_matcher.matching()
