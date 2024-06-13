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


class FIR(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.L_textbox = None
        self.Resampling_button = None
        self.M_textbox = None
        self.table_view = None
        self.model = None
        self.Filter_Convolove_button = None
        self.Filter_button = None
        self.values = None
        self.indexes = None
        self.num_samples = None
        self.is_periodic = None
        self.Signal = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('FIR')
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(200, 0, self.width(), self.height())
        self.Filter_button = QtWidgets.QPushButton('Make Filter', self)
        self.Filter_button.move(200, 350)
        self.Filter_button.clicked.connect(self.Make_filter)
        self.Filter_Convolove_button = QtWidgets.QPushButton('Filter Convolove', self)
        self.Filter_Convolove_button.move(350, 350)
        self.Filter_Convolove_button.clicked.connect(self.Convolove_filtter)
        self.Resampling_button = QtWidgets.QPushButton('Resampling ', self)
        self.Resampling_button.move(500, 350)
        self.Resampling_button.clicked.connect(self.Resampling)
        self.M_textbox = QtWidgets.QLineEdit(self)
        self.M_textbox.setPlaceholderText("Enter M")
        self.M_textbox.move(650, 350)
        self.L_textbox = QtWidgets.QLineEdit(self)
        self.L_textbox.setPlaceholderText("Enter L")
        self.L_textbox.move(650, 300)
        self.model = QtGui.QStandardItemModel(self)
        self.model.setHorizontalHeaderLabels(['Filter Type', 'Fs', 'Stop Band', 'Transition Band', 'F1', 'F2'])
        self.table_view = QtWidgets.QTableView(self)
        self.table_view.setModel(self.model)
        self.table_view.setGeometry(10, 390, 800, 150)

    def preprossing(self, Path):
        with open(Path, 'r') as f:
            self.Signal = int(f.readline().strip())
            self.is_periodic = int(f.readline().strip())
            self.num_samples = int(f.readline().strip())
            samples = [list(map(float, line.strip().split())) for line in f]
            self.indexes = [sample[0] for sample in samples]
            self.values = [sample[1] for sample in samples]
            return self.indexes, self.values

    def saveCoefficientsToFile(self, x_values, result_f):
        output_file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Coefficients", "",
                                                                    "Text Files (*.txt);;All Files (*)")
        if output_file_path:
            with open(output_file_path, 'w') as file:
                for x, coefficient in zip(x_values, result_f):
                    file.write(f"{x}\t{coefficient}\n")
        return output_file_path

    @staticmethod
    def next_odd_num(num=0.0):
        if num.__ceil__() % 2 == 0:
            return num.__ceil__() + 1
        elif num.__ceil__() % 1 == 0:
            return num.__ceil__()

    @staticmethod
    def read_filter_parameters(file_path):
        parameters = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(' = ')
                parameters[key] = value
        f2 = None
        filter_type = parameters['FilterType']
        fs = int(parameters['FS'])
        stop_band = int(parameters['StopBandAttenuation'])
        transition_band = int(parameters['TransitionBand'])
        if filter_type.__contains__('Band'):
            f1 = int(parameters['F1'])
            f2 = int(parameters['F2'])
        else:
            f1 = int(parameters['FC'])
        return filter_type, fs, stop_band, transition_band, f1, f2

    def Make_filter(self):
        global type_of_window, total_elements, total_elements, x_f_values3, y_f_values3
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            filter_type, fs, stop_band, transition_band, f1, f2 = self.read_filter_parameters(file_path)
            print(filter_type, fs, stop_band, transition_band, f1, f2)
            self.model.clear()
            self.model.appendRow([
                QtGui.QStandardItem(str(filter_type)),
                QtGui.QStandardItem(str(fs)),
                QtGui.QStandardItem(str(stop_band)),
                QtGui.QStandardItem(str(transition_band)),
                QtGui.QStandardItem(str(f1)),
                QtGui.QStandardItem(str(f2))
            ])
            bands = [21, 44, 53, 74]
            window_names = ['rectangular', 'hanning', 'hamming', 'blackman']
            for i in range(len(bands)):
                if bands[i] >= stop_band:
                    print(window_names[i])
                    type_of_window = window_names[i]
                    break

            print(type_of_window)
            normalized_transition_band = transition_band / fs
            if type_of_window == 'rectangular':
                total_elements = self.next_odd_num(0.9 / normalized_transition_band)
                print(total_elements)
            elif type_of_window == 'hanning':
                total_elements = self.next_odd_num(3.1 / normalized_transition_band)
                print(total_elements)
            elif type_of_window == 'hamming':
                total_elements = self.next_odd_num(3.3 / normalized_transition_band)
                print(total_elements)
            else:
                total_elements = self.next_odd_num(5.5 / normalized_transition_band)
                print(total_elements)
            if filter_type == 'Low pass':
                new_fc = (f1 + (transition_band / 2)) / fs, None
                print(new_fc)
            elif filter_type == 'High pass':
                new_fc = (f1 - (transition_band / 2)) / fs, None
                print()
            elif filter_type == 'Band pass':
                new_fc = (f1 - (transition_band / 2)) / fs, (f2 + (transition_band / 2)) / fs
                print(new_fc)
            else:
                new_fc = (f1 + (transition_band / 2)) / fs, (f2 - (transition_band / 2)) / fs
                print(new_fc)
            list_1 = []
            list_2 = []
            for i in range((total_elements / 2).__ceil__()):
                list_1.append(i)
                list_2.append(-i)
            indicates = list_1 + list_2
            indicates = list(set(indicates))
            indicates.sort()
            print(indicates)
            windows_list = []
            for element_index in range((total_elements / 2).__ceil__()):
                if type_of_window == 'hanning':
                    windows_list.append(0.5 + (0.5 * math.cos((2 * math.pi * element_index) / total_elements)))
                elif type_of_window == 'hamming':
                    windows_list.append(0.54 + (0.46 * math.cos((2 * math.pi * element_index) / total_elements)))
                elif type_of_window == 'blackman':
                    first_element = 0.5 * math.cos((2 * math.pi * element_index) / (total_elements - 1))
                    second_element = 0.08 * math.cos((4 * math.pi * element_index) / (total_elements - 1))
                    windows_list.append(0.42 + second_element + first_element)
            print(windows_list)
            f1 = new_fc[0]
            f2 = new_fc[1]
            filtered_list = []
            for element_index in range((total_elements / 2).__ceil__()):
                if filter_type == 'Low pass':
                    if element_index == 0:
                        filtered_list.append(2 * f1)
                    else:
                        x = element_index * 2 * math.pi * f1
                        filtered_list.append(2 * f1 * (math.sin(x) / x))
                elif filter_type == 'High pass':
                    if element_index == 0:
                        filtered_list.append(1 - (2 * f1))
                    else:
                        x = element_index * 2 * math.pi * f1
                        filtered_list.append(-2 * f1 * (math.sin(x) / x))
                elif filter_type == 'Band pass':
                    if element_index == 0:
                        filtered_list.append(2 * round(f2 - f1, 2))
                    else:
                        x_1 = element_index * 2 * math.pi * f1
                        x_2 = element_index * 2 * math.pi * f2
                        filtered_list.append((2 * f2 * (math.sin(x_2) / x_2)) - (2 * f1 * (math.sin(x_1) / x_1)))
                else:
                    if element_index == 0:
                        filtered_list.append(1 - (2 * (f2 - f1)))
                    else:
                        x__1 = element_index * 2 * math.pi * f1
                        x__2 = element_index * 2 * math.pi * f2
                        filtered_list.append(
                            ((2 * f2 * (math.sin(x__2) / x__2)) - (2 * f1 * (math.sin(x__1) / x__1))) * -1)
            print(filtered_list)
            list1 = [x * y for x, y in zip(windows_list, filtered_list)]
            list2 = [x * y for x, y in zip(windows_list, filtered_list)]
            list2.reverse()
            list2.extend(list1)
            list2.remove(list2[int(len(list2) / 2)])
            print(list2)
            file_dialog = QtWidgets.QFileDialog()
            file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
            file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                x_f_values3, y_f_values3 = self.preprossing(file_path)
            test_message = FIR_test.Compare_Signals(file_path, indicates, list2)
            self.saveCoefficientsToFile(indicates, list2)
            QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            axs[0].stem(x_f_values3, y_f_values3, linefmt='b-', markerfmt='bo', basefmt='r-')
            axs[0].set_title('Result - Discrete Plot')
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            axs[0].grid(True)
            axs[1].plot(x_f_values3, y_f_values3, 'r-')
            axs[1].scatter(x_f_values3, y_f_values3, color='red', marker='o')
            axs[1].set_title('Result - Continuous Plot')
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Y')
            axs[1].grid(True)
            plt.tight_layout()
            plt.show()

    def Convolove_filtter(self):
        global type_of_window, total_elements, total_elements, x_f_values1, x_f_values2, y_f_values2
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            filter_type, fs, stop_band, transition_band, f1, f2 = self.read_filter_parameters(file_path)
            print(filter_type, fs, stop_band, transition_band, f1, f2)
            self.model.clear()
            self.model.appendRow([
                QtGui.QStandardItem(str(filter_type)),
                QtGui.QStandardItem(str(fs)),
                QtGui.QStandardItem(str(stop_band)),
                QtGui.QStandardItem(str(transition_band)),
                QtGui.QStandardItem(str(f1)),
                QtGui.QStandardItem(str(f2))
            ])
            bands = [21, 44, 53, 74]
            window_names = ['rectangular', 'hanning', 'hamming', 'blackman']
            for i in range(len(bands)):
                if bands[i] >= stop_band:
                    print(window_names[i])
                    type_of_window = window_names[i]
                    break

            print(type_of_window)
            normalized_transition_band = transition_band / fs
            if type_of_window == 'rectangular':
                total_elements = self.next_odd_num(0.9 / normalized_transition_band)
                print(total_elements)
            elif type_of_window == 'hanning':
                total_elements = self.next_odd_num(3.1 / normalized_transition_band)
                print(total_elements)
            elif type_of_window == 'hamming':
                total_elements = self.next_odd_num(3.3 / normalized_transition_band)
                print(total_elements)
            else:
                total_elements = self.next_odd_num(5.5 / normalized_transition_band)
                print(total_elements)
            if filter_type == 'Low pass':
                new_fc = (f1 + (transition_band / 2)) / fs, None
                print(new_fc)
            elif filter_type == 'High pass':
                new_fc = (f1 - (transition_band / 2)) / fs, None
                print()
            elif filter_type == 'Band pass':
                new_fc = (f1 - (transition_band / 2)) / fs, (f2 + (transition_band / 2)) / fs
                print(new_fc)
            else:
                new_fc = (f1 + (transition_band / 2)) / fs, (f2 - (transition_band / 2)) / fs
                print(new_fc)
            list_1 = []
            list_2 = []
            for i in range((total_elements / 2).__ceil__()):
                list_1.append(i)
                list_2.append(-i)
            indicates = list_1 + list_2
            indicates = list(set(indicates))
            indicates.sort()
            print(indicates)
            windows_list = []
            for element_index in range((total_elements / 2).__ceil__()):
                if type_of_window == 'hanning':
                    windows_list.append(0.5 + (0.5 * math.cos((2 * math.pi * element_index) / total_elements)))
                elif type_of_window == 'hamming':
                    windows_list.append(0.54 + (0.46 * math.cos((2 * math.pi * element_index) / total_elements)))
                elif type_of_window == 'blackman':
                    first_element = 0.5 * math.cos((2 * math.pi * element_index) / (total_elements - 1))
                    second_element = 0.08 * math.cos((4 * math.pi * element_index) / (total_elements - 1))
                    windows_list.append(0.42 + second_element + first_element)
            print(windows_list)
            f1 = new_fc[0]
            f2 = new_fc[1]
            filtered_list = []
            for element_index in range((total_elements / 2).__ceil__()):
                if filter_type == 'Low pass':
                    if element_index == 0:
                        filtered_list.append(2 * f1)
                    else:
                        x = element_index * 2 * math.pi * f1
                        filtered_list.append(2 * f1 * (math.sin(x) / x))
                elif filter_type == 'High pass':
                    if element_index == 0:
                        filtered_list.append(1 - (2 * f1))
                    else:
                        x = element_index * 2 * math.pi * f1
                        filtered_list.append(-2 * f1 * (math.sin(x) / x))
                elif filter_type == 'Band pass':
                    if element_index == 0:
                        filtered_list.append(2 * round(f2 - f1, 2))
                    else:
                        x_1 = element_index * 2 * math.pi * f1
                        x_2 = element_index * 2 * math.pi * f2
                        filtered_list.append((2 * f2 * (math.sin(x_2) / x_2)) - (2 * f1 * (math.sin(x_1) / x_1)))
                else:
                    if element_index == 0:
                        filtered_list.append(1 - (2 * (f2 - f1)))
                    else:
                        x__1 = element_index * 2 * math.pi * f1
                        x__2 = element_index * 2 * math.pi * f2
                        filtered_list.append(
                            ((2 * f2 * (math.sin(x__2) / x__2)) - (2 * f1 * (math.sin(x__1) / x__1))) * -1)
            print(filtered_list)
            list1 = [x * y for x, y in zip(windows_list, filtered_list)]
            list2 = [x * y for x, y in zip(windows_list, filtered_list)]
            list2.reverse()
            list2.extend(list1)
            list2.remove(list2[int(len(list2) / 2)])
            print(list2)
            file_dialog = QtWidgets.QFileDialog()
            file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
            file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                x_f_values1, y_f_values1 = self.preprossing(file_path)
                print(x_f_values1, y_f_values1)
                len_signal_1 = len(list2)
                len_signal_2 = len(y_f_values1)
                len_output_signal = len_signal_1 + len_signal_2 - 1
                output_signal = [0] * len_output_signal
                for n in range(len_output_signal):
                    for k in range(max(0, n - len_signal_2 + 1), min(len_signal_1, n + 1)):
                        output_signal[n] += list2[k] * y_f_values1[n - k]
                output_indexes = indicates + x_f_values1
                x = list(set(output_indexes))
                x.sort()
                print(x, output_signal)
                print(len(x), len(output_signal))
                file_dialog = QtWidgets.QFileDialog()
                file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
                file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
                if file_dialog.exec_():
                    file_path = file_dialog.selectedFiles()[0]
                    x_f_values2, y_f_values2 = self.preprossing(file_path)
                test_message = FIR_test.Compare_Signals(file_path, x, output_signal)
                self.saveCoefficientsToFile(x, output_signal)
                QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
                fig, axs = plt.subplots(2, 1, figsize=(8, 8))
                axs[0].stem(x_f_values2, y_f_values2, linefmt='b-', markerfmt='bo', basefmt='r-')
                axs[0].set_title('Result - Discrete Plot')
                axs[0].set_xlabel('X')
                axs[0].set_ylabel('Y')
                axs[0].grid(True)
                axs[1].plot(x_f_values2, y_f_values2, 'r-')
                axs[1].scatter(x_f_values2, y_f_values2, color='red', marker='o')
                axs[1].set_title('Result - Continuous Plot')
                axs[1].set_xlabel('X')
                axs[1].set_ylabel('Y')
                axs[1].grid(True)
                plt.tight_layout()
                plt.show()

    def Resampling(self):
        value = self.M_textbox.text()
        try:
            M = int(value)
            print(M)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for M.')
            return
        value = self.L_textbox.text()
        try:
            L = int(value)
            print(L)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for L.')
            return
        # up
        if M == 0 and L != 0:
            file_dialog = QtWidgets.QFileDialog()
            file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
            file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)

            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                X_UP, Y_UP = self.preprossing(file_path)
                print(X_UP, Y_UP)
                upsampled_signal = []
                factor = 3
                for value in Y_UP:
                    upsampled_signal.extend([value] + [0] * (factor - 1))

                global type_of_window, total_elements, total_elements, x_f_values1, x_f_values2, y_f_values2, x_f_values4, y_f_values4
                file_dialog = QtWidgets.QFileDialog()
                file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
                file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
                if file_dialog.exec_():
                    file_path = file_dialog.selectedFiles()[0]
                    filter_type, fs, stop_band, transition_band, f1, f2 = self.read_filter_parameters(file_path)
                    print(filter_type, fs, stop_band, transition_band, f1, f2)
                    self.model.clear()
                    self.model.appendRow([
                        QtGui.QStandardItem(str(filter_type)),
                        QtGui.QStandardItem(str(fs)),
                        QtGui.QStandardItem(str(stop_band)),
                        QtGui.QStandardItem(str(transition_band)),
                        QtGui.QStandardItem(str(f1)),
                        QtGui.QStandardItem(str(f2))
                    ])
                    bands = [21, 44, 53, 74]
                    window_names = ['rectangular', 'hanning', 'hamming', 'blackman']
                    for i in range(len(bands)):
                        if bands[i] >= stop_band:
                            print(window_names[i])
                            type_of_window = window_names[i]
                            break

                    print(type_of_window)
                    normalized_transition_band = transition_band / fs
                    if type_of_window == 'rectangular':
                        total_elements = self.next_odd_num(0.9 / normalized_transition_band)
                        print(total_elements)
                    elif type_of_window == 'hanning':
                        total_elements = self.next_odd_num(3.1 / normalized_transition_band)
                        print(total_elements)
                    elif type_of_window == 'hamming':
                        total_elements = self.next_odd_num(3.3 / normalized_transition_band)
                        print(total_elements)
                    else:
                        total_elements = self.next_odd_num(5.5 / normalized_transition_band)
                        print(total_elements)
                    if filter_type == 'Low pass':
                        new_fc = (f1 + (transition_band / 2)) / fs, None
                        print(new_fc)
                    elif filter_type == 'High pass':
                        new_fc = (f1 - (transition_band / 2)) / fs, None
                        print()
                    elif filter_type == 'Band pass':
                        new_fc = (f1 - (transition_band / 2)) / fs, (f2 + (transition_band / 2)) / fs
                        print(new_fc)
                    else:
                        new_fc = (f1 + (transition_band / 2)) / fs, (f2 - (transition_band / 2)) / fs
                        print(new_fc)
                    list_1 = []
                    list_2 = []
                    for i in range((total_elements / 2).__ceil__()):
                        list_1.append(i)
                        list_2.append(-i)
                    indicates = list_1 + list_2
                    indicates = list(set(indicates))
                    indicates.sort()
                    print(indicates)
                    windows_list = []
                    for element_index in range((total_elements / 2).__ceil__()):
                        if type_of_window == 'hanning':
                            windows_list.append(0.5 + (0.5 * math.cos((2 * math.pi * element_index) / total_elements)))
                        elif type_of_window == 'hamming':
                            windows_list.append(
                                0.54 + (0.46 * math.cos((2 * math.pi * element_index) / total_elements)))
                        elif type_of_window == 'blackman':
                            first_element = 0.5 * math.cos((2 * math.pi * element_index) / (total_elements - 1))
                            second_element = 0.08 * math.cos((4 * math.pi * element_index) / (total_elements - 1))
                            windows_list.append(0.42 + second_element + first_element)
                    print(windows_list)
                    f1 = new_fc[0]
                    f2 = new_fc[1]
                    filtered_list = []
                    for element_index in range((total_elements / 2).__ceil__()):
                        if filter_type == 'Low pass':
                            if element_index == 0:
                                filtered_list.append(2 * f1)
                            else:
                                x = element_index * 2 * math.pi * f1
                                filtered_list.append(2 * f1 * (math.sin(x) / x))
                        elif filter_type == 'High pass':
                            if element_index == 0:
                                filtered_list.append(1 - (2 * f1))
                            else:
                                x = element_index * 2 * math.pi * f1
                                filtered_list.append(-2 * f1 * (math.sin(x) / x))
                        elif filter_type == 'Band pass':
                            if element_index == 0:
                                filtered_list.append(2 * round(f2 - f1, 2))
                            else:
                                x_1 = element_index * 2 * math.pi * f1
                                x_2 = element_index * 2 * math.pi * f2
                                filtered_list.append(
                                    (2 * f2 * (math.sin(x_2) / x_2)) - (2 * f1 * (math.sin(x_1) / x_1)))
                        else:
                            if element_index == 0:
                                filtered_list.append(1 - (2 * (f2 - f1)))
                            else:
                                x__1 = element_index * 2 * math.pi * f1
                                x__2 = element_index * 2 * math.pi * f2
                                filtered_list.append(
                                    ((2 * f2 * (math.sin(x__2) / x__2)) - (2 * f1 * (math.sin(x__1) / x__1))) * -1)
                    print(filtered_list)
                    list1 = [x * y for x, y in zip(windows_list, filtered_list)]
                    list2 = [x * y for x, y in zip(windows_list, filtered_list)]
                    list2.reverse()
                    list2.extend(list1)
                    list2.remove(list2[int(len(list2) / 2)])
                    print(list2)
                    len_signal_1 = len(list2)
                    len_signal_2 = len(upsampled_signal)
                    len_output_signal = len_signal_1 + len_signal_2 - 1
                    output_signal = [0] * len_output_signal
                    for n in range(len_output_signal):
                        for k in range(max(0, n - len_signal_2 + 1), min(len_signal_1, n + 1)):
                            output_signal[n] += list2[k] * upsampled_signal[n - k]
                    output_indexes = indicates + X_UP
                    x = list(set(output_indexes))
                    x.sort()
                    for i in range(2):
                        output_signal.remove(output_signal[-1])
                        x.remove(x[-1])
                    print(x, output_signal)
                    print(len(x), len(output_signal))

                    file_dialog = QtWidgets.QFileDialog()
                    file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
                    file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
                    if file_dialog.exec_():
                        file_path = file_dialog.selectedFiles()[0]
                        x_f_values2, y_f_values2 = self.preprossing(file_path)
                    test_message = FIR_test.Compare_Signals(file_path, x, output_signal)
                    self.saveCoefficientsToFile(x, output_signal)
                    QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
                    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
                    axs[0].stem(x_f_values2, y_f_values2, linefmt='b-', markerfmt='bo', basefmt='r-')
                    axs[0].set_title('Result - Discrete Plot')
                    axs[0].set_xlabel('X')
                    axs[0].set_ylabel('Y')
                    axs[0].grid(True)
                    axs[1].plot(x_f_values2, y_f_values2, 'r-')
                    axs[1].scatter(x_f_values2, x_f_values2, color='red', marker='o')
                    axs[1].set_title('Result - Continuous Plot')
                    axs[1].set_xlabel('X')
                    axs[1].set_ylabel('Y')
                    axs[1].grid(True)
                    plt.tight_layout()
                    plt.show()
        # down
        elif M != 0 and L == 0:

            file_dialog = QtWidgets.QFileDialog()
            file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
            file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                filter_type, fs, stop_band, transition_band, f1, f2 = self.read_filter_parameters(file_path)
                print(filter_type, fs, stop_band, transition_band, f1, f2)
                self.model.clear()
                self.model.appendRow([
                    QtGui.QStandardItem(str(filter_type)),
                    QtGui.QStandardItem(str(fs)),
                    QtGui.QStandardItem(str(stop_band)),
                    QtGui.QStandardItem(str(transition_band)),
                    QtGui.QStandardItem(str(f1)),
                    QtGui.QStandardItem(str(f2))
                ])
                bands = [21, 44, 53, 74]
                window_names = ['rectangular', 'hanning', 'hamming', 'blackman']
                for i in range(len(bands)):
                    if bands[i] >= stop_band:
                        print(window_names[i])
                        type_of_window = window_names[i]
                        break

                print(type_of_window)
                normalized_transition_band = transition_band / fs
                if type_of_window == 'rectangular':
                    total_elements = self.next_odd_num(0.9 / normalized_transition_band)
                    print(total_elements)
                elif type_of_window == 'hanning':
                    total_elements = self.next_odd_num(3.1 / normalized_transition_band)
                    print(total_elements)
                elif type_of_window == 'hamming':
                    total_elements = self.next_odd_num(3.3 / normalized_transition_band)
                    print(total_elements)
                else:
                    total_elements = self.next_odd_num(5.5 / normalized_transition_band)
                    print(total_elements)
                if filter_type == 'Low pass':
                    new_fc = (f1 + (transition_band / 2)) / fs, None
                    print(new_fc)
                elif filter_type == 'High pass':
                    new_fc = (f1 - (transition_band / 2)) / fs, None
                    print()
                elif filter_type == 'Band pass':
                    new_fc = (f1 - (transition_band / 2)) / fs, (f2 + (transition_band / 2)) / fs
                    print(new_fc)
                else:
                    new_fc = (f1 + (transition_band / 2)) / fs, (f2 - (transition_band / 2)) / fs
                    print(new_fc)
                list_1 = []
                list_2 = []
                for i in range((total_elements / 2).__ceil__()):
                    list_1.append(i)
                    list_2.append(-i)
                indicates = list_1 + list_2
                indicates = list(set(indicates))
                indicates.sort()
                print(indicates)
                windows_list = []
                for element_index in range((total_elements / 2).__ceil__()):
                    if type_of_window == 'hanning':
                        windows_list.append(0.5 + (0.5 * math.cos((2 * math.pi * element_index) / total_elements)))
                    elif type_of_window == 'hamming':
                        windows_list.append(
                            0.54 + (0.46 * math.cos((2 * math.pi * element_index) / total_elements)))
                    elif type_of_window == 'blackman':
                        first_element = 0.5 * math.cos((2 * math.pi * element_index) / (total_elements - 1))
                        second_element = 0.08 * math.cos((4 * math.pi * element_index) / (total_elements - 1))
                        windows_list.append(0.42 + second_element + first_element)
                print(windows_list)
                f1 = new_fc[0]
                f2 = new_fc[1]
                filtered_list = []
                for element_index in range((total_elements / 2).__ceil__()):
                    if filter_type == 'Low pass':
                        if element_index == 0:
                            filtered_list.append(2 * f1)
                        else:
                            x = element_index * 2 * math.pi * f1
                            filtered_list.append(2 * f1 * (math.sin(x) / x))
                    elif filter_type == 'High pass':
                        if element_index == 0:
                            filtered_list.append(1 - (2 * f1))
                        else:
                            x = element_index * 2 * math.pi * f1
                            filtered_list.append(-2 * f1 * (math.sin(x) / x))
                    elif filter_type == 'Band pass':
                        if element_index == 0:
                            filtered_list.append(2 * round(f2 - f1, 2))
                        else:
                            x_1 = element_index * 2 * math.pi * f1
                            x_2 = element_index * 2 * math.pi * f2
                            filtered_list.append(
                                (2 * f2 * (math.sin(x_2) / x_2)) - (2 * f1 * (math.sin(x_1) / x_1)))
                    else:
                        if element_index == 0:
                            filtered_list.append(1 - (2 * (f2 - f1)))
                        else:
                            x__1 = element_index * 2 * math.pi * f1
                            x__2 = element_index * 2 * math.pi * f2
                            filtered_list.append(
                                ((2 * f2 * (math.sin(x__2) / x__2)) - (2 * f1 * (math.sin(x__1) / x__1))) * -1)
                print(filtered_list)
                list1 = [x * y for x, y in zip(windows_list, filtered_list)]
                list2 = [x * y for x, y in zip(windows_list, filtered_list)]
                list2.reverse()
                list2.extend(list1)
                list2.remove(list2[int(len(list2) / 2)])
                print(list2)
                file_dialog = QtWidgets.QFileDialog()
                file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
                file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
                if file_dialog.exec_():
                    file_path = file_dialog.selectedFiles()[0]
                    x_f_values1, y_f_values1 = self.preprossing(file_path)
                    print(x_f_values1, y_f_values1)
                    len_signal_1 = len(list2)
                    len_signal_2 = len(y_f_values1)
                    len_output_signal = len_signal_1 + len_signal_2 - 1
                    output_signal = [0] * len_output_signal
                    for n in range(len_output_signal):
                        for k in range(max(0, n - len_signal_2 + 1), min(len_signal_1, n + 1)):
                            output_signal[n] += list2[k] * y_f_values1[n - k]
                    output_indexes = indicates + x_f_values1
                    x = list(set(output_indexes))
                    x.sort()
                    print(x, output_signal)
                    print(len(x), len(output_signal))

                    down_sampled_signal = []
                    down_sampled_signal_indicates = []
                    for i in range(0, len(output_signal), M):
                        down_sampled_signal.append(output_signal[i])
                    for i in range(len(down_sampled_signal)):
                        down_sampled_signal_indicates.append(x[i])
                    print(len(down_sampled_signal_indicates))
                    print(len(down_sampled_signal))
                    print(down_sampled_signal_indicates, down_sampled_signal)
                    file_dialog = QtWidgets.QFileDialog()
                    file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
                    file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
                    if file_dialog.exec_():
                        file_path = file_dialog.selectedFiles()[0]
                        x_f_values4, y_f_values4 = self.preprossing(file_path)
                    test_message = FIR_test.Compare_Signals(file_path, down_sampled_signal_indicates,
                                                            down_sampled_signal)
                    self.saveCoefficientsToFile(down_sampled_signal_indicates, down_sampled_signal)
                    QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
                    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
                    axs[0].stem(x_f_values4, y_f_values4, linefmt='b-', markerfmt='bo', basefmt='r-')
                    axs[0].set_title('Result - Discrete Plot')
                    axs[0].set_xlabel('X')
                    axs[0].set_ylabel('Y')
                    axs[0].grid(True)
                    axs[1].plot(x_f_values4, y_f_values4, 'r-')
                    axs[1].scatter(x_f_values4, y_f_values4, color='red', marker='o')
                    axs[1].set_title('Result - Continuous Plot')
                    axs[1].set_xlabel('X')
                    axs[1].set_ylabel('Y')
                    axs[1].grid(True)
                    plt.tight_layout()
                    plt.show()
        # up and down
        elif M != 0 and L != 0:
            file_dialog = QtWidgets.QFileDialog()
            file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
            file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)

            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                X_UP, Y_UP = self.preprossing(file_path)
                print(X_UP, Y_UP)
                upsampled_signal = []
                factor = 3
                for value in Y_UP:
                    upsampled_signal.extend([value] + [0] * (factor - 1))

                file_dialog = QtWidgets.QFileDialog()
                file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
                file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
                if file_dialog.exec_():
                    file_path = file_dialog.selectedFiles()[0]
                    filter_type, fs, stop_band, transition_band, f1, f2 = self.read_filter_parameters(file_path)
                    print(filter_type, fs, stop_band, transition_band, f1, f2)
                    self.model.clear()
                    self.model.appendRow([
                        QtGui.QStandardItem(str(filter_type)),
                        QtGui.QStandardItem(str(fs)),
                        QtGui.QStandardItem(str(stop_band)),
                        QtGui.QStandardItem(str(transition_band)),
                        QtGui.QStandardItem(str(f1)),
                        QtGui.QStandardItem(str(f2))
                    ])
                    bands = [21, 44, 53, 74]
                    window_names = ['rectangular', 'hanning', 'hamming', 'blackman']
                    for i in range(len(bands)):
                        if bands[i] >= stop_band:
                            print(window_names[i])
                            type_of_window = window_names[i]
                            break

                    print(type_of_window)
                    normalized_transition_band = transition_band / fs
                    if type_of_window == 'rectangular':
                        total_elements = self.next_odd_num(0.9 / normalized_transition_band)
                        print(total_elements)
                    elif type_of_window == 'hanning':
                        total_elements = self.next_odd_num(3.1 / normalized_transition_band)
                        print(total_elements)
                    elif type_of_window == 'hamming':
                        total_elements = self.next_odd_num(3.3 / normalized_transition_band)
                        print(total_elements)
                    else:
                        total_elements = self.next_odd_num(5.5 / normalized_transition_band)
                        print(total_elements)
                    if filter_type == 'Low pass':
                        new_fc = (f1 + (transition_band / 2)) / fs, None
                        print(new_fc)
                    elif filter_type == 'High pass':
                        new_fc = (f1 - (transition_band / 2)) / fs, None
                        print()
                    elif filter_type == 'Band pass':
                        new_fc = (f1 - (transition_band / 2)) / fs, (f2 + (transition_band / 2)) / fs
                        print(new_fc)
                    else:
                        new_fc = (f1 + (transition_band / 2)) / fs, (f2 - (transition_band / 2)) / fs
                        print(new_fc)
                    list_1 = []
                    list_2 = []
                    for i in range((total_elements / 2).__ceil__()):
                        list_1.append(i)
                        list_2.append(-i)
                    indicates = list_1 + list_2
                    indicates = list(set(indicates))
                    indicates.sort()
                    print(indicates)
                    windows_list = []
                    for element_index in range((total_elements / 2).__ceil__()):
                        if type_of_window == 'hanning':
                            windows_list.append(0.5 + (0.5 * math.cos((2 * math.pi * element_index) / total_elements)))
                        elif type_of_window == 'hamming':
                            windows_list.append(
                                0.54 + (0.46 * math.cos((2 * math.pi * element_index) / total_elements)))
                        elif type_of_window == 'blackman':
                            first_element = 0.5 * math.cos((2 * math.pi * element_index) / (total_elements - 1))
                            second_element = 0.08 * math.cos((4 * math.pi * element_index) / (total_elements - 1))
                            windows_list.append(0.42 + second_element + first_element)
                    print(windows_list)
                    f1 = new_fc[0]
                    f2 = new_fc[1]
                    filtered_list = []
                    for element_index in range((total_elements / 2).__ceil__()):
                        if filter_type == 'Low pass':
                            if element_index == 0:
                                filtered_list.append(2 * f1)
                            else:
                                x = element_index * 2 * math.pi * f1
                                filtered_list.append(2 * f1 * (math.sin(x) / x))
                        elif filter_type == 'High pass':
                            if element_index == 0:
                                filtered_list.append(1 - (2 * f1))
                            else:
                                x = element_index * 2 * math.pi * f1
                                filtered_list.append(-2 * f1 * (math.sin(x) / x))
                        elif filter_type == 'Band pass':
                            if element_index == 0:
                                filtered_list.append(2 * round(f2 - f1, 2))
                            else:
                                x_1 = element_index * 2 * math.pi * f1
                                x_2 = element_index * 2 * math.pi * f2
                                filtered_list.append(
                                    (2 * f2 * (math.sin(x_2) / x_2)) - (2 * f1 * (math.sin(x_1) / x_1)))
                        else:
                            if element_index == 0:
                                filtered_list.append(1 - (2 * (f2 - f1)))
                            else:
                                x__1 = element_index * 2 * math.pi * f1
                                x__2 = element_index * 2 * math.pi * f2
                                filtered_list.append(
                                    ((2 * f2 * (math.sin(x__2) / x__2)) - (2 * f1 * (math.sin(x__1) / x__1))) * -1)
                    print(filtered_list)
                    list1 = [x * y for x, y in zip(windows_list, filtered_list)]
                    list2 = [x * y for x, y in zip(windows_list, filtered_list)]
                    list2.reverse()
                    list2.extend(list1)
                    list2.remove(list2[int(len(list2) / 2)])
                    print(list2)
                    len_signal_1 = len(list2)
                    len_signal_2 = len(upsampled_signal)
                    len_output_signal = len_signal_1 + len_signal_2 - 1
                    output_signal = [0] * len_output_signal
                    for n in range(len_output_signal):
                        for k in range(max(0, n - len_signal_2 + 1), min(len_signal_1, n + 1)):
                            output_signal[n] += list2[k] * upsampled_signal[n - k]
                    output_indexes = indicates + X_UP
                    x = list(set(output_indexes))
                    x.sort()

                    print(x, output_signal)
                    print(len(x), len(output_signal))
                    down_sampled_signal = []
                    down_sampled_signal_indicates = []

                    for i in range(0, len(output_signal), M):
                        print(output_signal[i])
                        down_sampled_signal.append(output_signal[i])
                    print(len(down_sampled_signal))

                    # Corrected loop
                    for i in range(min(len(down_sampled_signal), len(x))):
                        down_sampled_signal_indicates.append(x[i])

                    print(len(down_sampled_signal_indicates))
                    print(len(down_sampled_signal))
                    print(down_sampled_signal_indicates, down_sampled_signal)
                    for i in range(1):
                        down_sampled_signal.remove(down_sampled_signal[-1])
                        down_sampled_signal_indicates.remove(down_sampled_signal_indicates[-1])
                    file_dialog = QtWidgets.QFileDialog()
                    file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
                    file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
                    if file_dialog.exec_():
                        file_path = file_dialog.selectedFiles()[0]
                        x_f_values4, y_f_values4 = self.preprossing(file_path)
                    test_message = FIR_test.Compare_Signals(file_path, down_sampled_signal_indicates,
                                                            down_sampled_signal)
                    self.saveCoefficientsToFile(down_sampled_signal_indicates, down_sampled_signal)
                    QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
                    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
                    axs[0].stem(x_f_values4, y_f_values4, linefmt='b-', markerfmt='bo', basefmt='r-')
                    axs[0].set_title('Result - Discrete Plot')
                    axs[0].set_xlabel('X')
                    axs[0].set_ylabel('Y')
                    axs[0].grid(True)
                    axs[1].plot(x_f_values4, y_f_values4, 'r-')
                    axs[1].scatter(x_f_values4, y_f_values4, color='red', marker='o')
                    axs[1].set_title('Result - Continuous Plot')
                    axs[1].set_xlabel('X')
                    axs[1].set_ylabel('Y')
                    axs[1].grid(True)
                    plt.tight_layout()
                    plt.show()

        else:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for M and L.')
