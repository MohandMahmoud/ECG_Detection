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


class Operations(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.values_two = None
        self.indexes_two = None
        self.num_samples_two = None
        self.is_periodic_two = None
        self.signal_two_type = None
        self.values_one = None
        self.indexes_one = None
        self.num_samples_one = None
        self.is_periodic_one = None
        self.signal_one_type = None
        self.input_files = None
        self.norm_option_neg_1_1 = None
        self.norm_option_0_1 = None
        self.norm_option_label = None
        self.constant_input_shifting = None
        self.constant_label_shifting = None
        self.constant_input_multiplication = None
        self.constant_label_multiplication = None
        self.input_files_list = None
        self.operation_combo = None
        self.output_data = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Arithmetic Operations')
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(380, 0, self.width(), self.height())
        label = QtWidgets.QLabel('Arithmetic Operations:', self)
        label.move(50, 50)
        operation_label = QtWidgets.QLabel('Operation:', self)
        operation_label.move(50, 100)
        self.operation_combo = QtWidgets.QComboBox(self)
        self.operation_combo.addItems(
            ['Addition', 'Subtraction', 'Multiplication', 'Squaring', 'Shifting', 'Normalization', 'Accumulation'])
        self.operation_combo.setCurrentIndex(0)
        self.operation_combo.move(200, 100)
        browse_input_btn1 = QtWidgets.QPushButton('Browse Input File(1)', self)
        browse_input_btn1.clicked.connect(self.browse_input1)
        browse_input_btn1.move(50, 150)
        browse_input_btn2 = QtWidgets.QPushButton('Browse Input File(2)', self)
        browse_input_btn2.clicked.connect(self.browse_input2)
        browse_input_btn2.move(180, 150)
        perform_operation_btn = QtWidgets.QPushButton('Perform Operation', self)
        perform_operation_btn.clicked.connect(self.perform_operation)
        perform_operation_btn.move(50, 200)
        display_continuous_signal_btn = QtWidgets.QPushButton('Display Continuous Signal', self)
        display_continuous_signal_btn.clicked.connect(self.display_continuous_signal)
        display_continuous_signal_btn.move(50, 250)
        display_discrete_signal_btn = QtWidgets.QPushButton('Display Discrete Signal', self)
        display_discrete_signal_btn.clicked.connect(self.display_discrete_signal)
        display_discrete_signal_btn.move(50, 300)
        input_files_label = QtWidgets.QLabel('Selected Input Files:', self)
        input_files_label.move(50, 350)
        self.input_files_list = QtWidgets.QListWidget(self)
        self.input_files_list.setGeometry(50, 370, 250, 100)
        self.constant_label_multiplication = QtWidgets.QLabel('Constant (for Multiplication):', self)
        self.constant_label_multiplication.move(350, 50)
        self.constant_input_multiplication = QtWidgets.QLineEdit(self)
        self.constant_input_multiplication.setGeometry(520, 45, 100, 30)
        self.constant_label_shifting = QtWidgets.QLabel('Constant (for Shifting):', self)
        self.constant_label_shifting.move(380, 85)
        self.constant_input_shifting = QtWidgets.QLineEdit(self)
        self.constant_input_shifting.setGeometry(520, 80, 100, 30)
        self.norm_option_label = QtWidgets.QLabel('Normalization Option:', self)
        self.norm_option_label.move(50, 500)
        self.norm_option_0_1 = QtWidgets.QRadioButton('0-1', self)
        self.norm_option_0_1.move(400, 130)
        self.norm_option_0_1.setChecked(True)
        self.norm_option_neg_1_1 = QtWidgets.QRadioButton('-1-1', self)
        self.norm_option_neg_1_1.move(480, 130)
        self.input_files = []
        self.output_data = None

    def get_selected_operation(self):
        return self.operation_combo.currentText()

    def browse_input1(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose First Signal File", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'r') as file:
                self.signal_one_type = int(file.readline().strip())
                self.is_periodic_one = int(file.readline().strip())
                self.num_samples_one = int(file.readline().strip())
                samples_one = [list(map(float, line.strip().split())) for line in file]
                self.indexes_one = [sample[0] for sample in samples_one]
                self.values_one = [sample[1] for sample in samples_one]
                print(self.values_one)
                self.input_files.append(self.values_one)

    def browse_input2(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose Second Signal File", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'r') as file:
                self.signal_two_type = int(file.readline().strip())
                self.is_periodic_two = int(file.readline().strip())
                self.num_samples_two = int(file.readline().strip())
                samples_two = [list(map(float, line.strip().split())) for line in file]
                self.indexes_two = [sample[0] for sample in samples_two]
                self.values_two = [sample[1] for sample in samples_two]
                print(self.values_two)
                self.input_files.append(self.values_two)

    def get_constant_multiplication(self):
        constant = self.constant_input_multiplication.text()
        if constant:
            try:
                return float(constant)
            except ValueError:
                QtWidgets.QMessageBox.information(self, 'Error', 'Invalid constant value for Multiplication.')
        return None

    def get_constant_shifting(self):
        constant = self.constant_input_shifting.text()
        if constant:
            try:
                return float(constant)
            except ValueError:
                QtWidgets.QMessageBox.information(self, 'Error', 'Invalid constant value for Shifting.')
        return None

    def get_norm_option(self):
        if self.norm_option_0_1.isChecked():
            return '0-1'
        elif self.norm_option_neg_1_1.isChecked():
            return '-1-1'
        else:
            return ''

    def perform_operation(self):
        global result
        operation = self.get_selected_operation()
        constant_multiplication = self.get_constant_multiplication()
        constant_shifting = self.get_constant_shifting()

        if not self.input_files:
            QtWidgets.QMessageBox.information(self, 'Input Error', 'No input file(s) selected.')
            return

        try:
            if operation in ['Addition', 'Subtraction']:
                if len(self.input_files) != 2:
                    QtWidgets.QMessageBox.information(self, 'Input Error',
                                                      'Two input files required for Addition/Subtraction.')
                    return

                if operation == 'Addition':
                    result = self.add_signals()
                else:  # Subtraction
                    result = self.subtract_signals()
            else:
                if len(self.input_files) != 1:
                    QtWidgets.QMessageBox.information(self, 'Input Error',
                                                      'One input file required for this operation.')
                    return

                if operation == 'Multiplication':
                    if constant_multiplication is not None:
                        x = float(constant_multiplication)
                        result = self.multiply_signal(x)
                    else:
                        return
                elif operation == 'Squaring':
                    result = self.square_signal()
                elif operation == 'Shifting':
                    if constant_shifting is not None:
                        result = self.shift_signal(constant_shifting)
                    else:
                        return
                elif operation == 'Normalization':
                    result = self.normalize_signal()
                elif operation == 'Accumulation':
                    result = self.accumulate_signal()
                else:
                    QtWidgets.QMessageBox.information(self, 'Invalid Operation', 'Invalid operation selected.')

            self.output_data = result
            QtWidgets.QMessageBox.information(self, 'Operation Complete', 'Operation performed successfully.')

        except Exception as e:
            QtWidgets.QMessageBox.information(self, 'Error', f"An error occurred during {operation}: {e}")

    def add_signals(self):
        x = self.values_one + self.values_two
        return x

    def subtract_signals(self):
        x = []
        for i in range(max(len(self.values_one), len(self.values_two))):
            t = self.values_one[i] - self.values_two[i]
            x.append(t)
        return x

    def multiply_signal(self, constant):
        x = []
        for i in range(len(self.values_one)):
            c = self.values_one[i] * constant
            x.append(c)
        return x

    def square_signal(self):
        x = []
        for i in range(len(self.values_one)):
            o = self.values_one[i] * self.values_one[i]
            x.append(o)
        return x

    def shift_signal(self, constant):
        x = []
        for i in range(len(self.indexes_one)):
            p = self.indexes_one[i] + constant
            x.append(p)
        return x

    def normalize_signal(self):
        data = self.values_one
        norm_option = self.get_norm_option()
        if norm_option == '0-1':
            i = (data - np.min(data)) / (np.max(data) - np.min(data))
        elif norm_option == '-1-1':
            i = (data - np.mean(data)) / np.max(np.abs(data))
        else:
            raise ValueError("Invalid normalization option")
        return i

    def accumulate_signal(self):
        x = []
        for i in range(len(self.values_one)):
            w = np.cumsum(self.values_one[i])
            x.append(w)
        return x

    def display_continuous_signal(self):
        if self.output_data is not None:
            plt.plot(self.output_data)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title('Continuous Signal')
            plt.grid(True)
            plt.show()
        else:
            QtWidgets.QMessageBox.information(self, 'Display Error', 'No data to display. Perform an operation first.')

    def display_discrete_signal(self):
        if self.output_data is not None:
            plt.stem(self.output_data, use_line_collection=True)
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Discrete Signal')
            plt.grid(True)
            plt.show()
        else:
            QtWidgets.QMessageBox.information(self, 'Display Error', 'No data to display. Perform an operation first.')
