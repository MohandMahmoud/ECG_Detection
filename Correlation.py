import math
import sys
import FIR_test
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import ConvTest
import Correlation_test
import DerivativeSignal
import Shift_Fold_Signal
from comparesignals import *
from QuanTest1 import *
from QuanTest2 import *
from scipy import signal
from scipy.fftpack import fft
import typing
import sys
import cmath
import DerivativeSignal as DS
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QTextEdit, \
    QFileDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from scipy.fftpack import dct
from Correlation_test import *
import os


class Correlation(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.m_textbox1 = None
        self.convolve_button2 = None
        self.Correlation_button = None
        self.values = None
        self.indexes = None
        self.num_samples = None
        self.is_periodic = None
        self.Signal = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Correlation')
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(200, 0, self.width(), self.height())
        self.convolve_button2 = QtWidgets.QPushButton('Correlation', self)
        self.convolve_button2.move(250, 350)
        self.convolve_button2.clicked.connect(self.normalized)
        self.convolve_button2 = QtWidgets.QPushButton('Time Analysis', self)
        self.convolve_button2.move(350, 350)
        self.convolve_button2.clicked.connect(self.Time)
        self.m_textbox1 = QtWidgets.QLineEdit(self)
        self.m_textbox1.setPlaceholderText("Enter the value of FS")
        self.m_textbox1.move(320, 400)
        self.convolve_button2 = QtWidgets.QPushButton('Matching', self)
        self.convolve_button2.move(450, 350)
        self.convolve_button2.clicked.connect(self.matching)

    def preprossing(self, Path):
        with open(Path, 'r') as f:
            self.Signal = int(f.readline().strip())
            self.is_periodic = int(f.readline().strip())
            self.num_samples = int(f.readline().strip())
            samples = [list(map(float, line.strip().split())) for line in f]
            self.indexes = [sample[0] for sample in samples]
            self.values = [sample[1] for sample in samples]
            return self.indexes, self.values

    def normalized(self):
        global Y_1, Y_2, X_1, r
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            X_1, Y_1 = self.preprossing(file_path)
            print(X_1, Y_1)

        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            X_2, Y_2 = self.preprossing(file_path)
            print(X_2, Y_2)

        sum_of_squares = sum([q ** 2 for q in Y_1])
        out = []

        def shift_left(lst):
            first_element = lst[0]
            shifted_lst = lst[1:] + [first_element]
            return shifted_lst

        Time_list = Y_2
        r = 0
        for i in range(len(Time_list)):
            for o in range(len(Time_list)):
                r += Y_1[o] * Time_list[o]
            r = (1 / 5) * r
            sum_of_squares_Y = sum([n ** 2 for n in Time_list])
            b = sum_of_squares * sum_of_squares_Y
            v = np.sqrt(b)
            t = (1 / 5) * v
            p = r / t
            out.append(p)
            shifted_list = shift_left(Time_list)
            Time_list = shifted_list
            r = 0
        print(X_1, out)
        test_message = Correlation_test.Compare_Signals("CorrOutput.txt", X_1, out)
        QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        axs[0].stem(X_1, out, linefmt='b-', markerfmt='bo', basefmt='r-')
        axs[0].set_title('Correlation Result - Discrete Plot')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].grid(True)
        axs[1].plot(X_1, out, 'r-')
        axs[1].scatter(X_1, out, color='red', marker='o')
        axs[1].set_title('Correlation Result - Continuous Plot')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()

    def Time(self):
        global Y_3, Y_4, original_list, X_4
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            X_3, Y_3 = self.preprossing(file_path)
            print(X_3, Y_3)

        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            X_4, Y_4 = self.preprossing(file_path)
            print(X_4, Y_4)

        value = self.m_textbox1.text()
        try:
            FS = int(value)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for m.')
            return
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
        max_corr_index = np.argmax(np.abs(out))
        lag_j = max_corr_index - len(Y_3)
        lag_j += 1
        TS = 1 / FS
        time_delay = max_corr_index * TS
        if time_delay == 5 / 100:
            QtWidgets.QMessageBox.information(self, 'Test Result',
                                              f"Test case is Successfully time delay is {time_delay}")

    @staticmethod
    def load_signals_from_folder(folder_path):
        signals = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                S = np.loadtxt(file_path)
                signals.append(S)
        return signals

    @staticmethod
    def coor(Y_6, Y_8):

        sum_of_squares = sum([q ** 2 for q in Y_6])
        out = []

        def shift_left(lst):
            first_element = lst[0]
            shifted_lst = lst[1:] + [first_element]
            return shifted_lst

        original = Y_8
        e = 0
        for i in range(len(original)):
            for o in range(len(original)):
                e += Y_6[o] * original[o]
            e = (1 / 5) * e
            sum_of_squares_Y = sum([n ** 2 for n in original])
            b = sum_of_squares * sum_of_squares_Y
            v = np.sqrt(b)
            t = (1 / 5) * v
            p = e / t
            out.append(p)
            shifted_list = shift_left(original)
            original = shifted_list
            e = 0
        return out

    def template_matching(self, test_signal, templates):
        correlations = self.coor(test_signal, templates)
        predicted_class = np.argmax(correlations)
        return predicted_class

    @staticmethod
    def compute_templates(class_signals):
        return [np.mean(Y, axis=0) for Y in class_signals]

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
            if predicted_class == 2:
                test_message = f"Test Signal {i + 1} Class 2 ⇒ up movement of EOG signal"

                QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
            else:
                test_message = f"Test Signal {i + 1} Class 1 ⇒ down movement of EOG signal"
                QtWidgets.QMessageBox.information(self, 'Test Result', test_message)