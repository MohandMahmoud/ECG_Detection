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


class Generate_Test(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.wave_type_combo = None
        self.amp_txt_box = None
        self.phase_shift_txt_box = None
        self.analog_freq_txt_box = None
        self.sampling_freq_txt_box = None
        self.selected_option = 'sine'
        self.amp = None
        self.phase_shift = None
        self.analog_freq = None
        self.sampling_freq = None
        self.x_axis, self.y_axis = None, None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Generate and Test Signals')
        self.setGeometry(100, 100, 800, 500)
        # Create a QLabel for the image
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)

        # Set the label's geometry to match the size of the main window
        image_label.setGeometry(380, 0, self.width(), self.height())

        label = QtWidgets.QLabel('Signal Generation:', self)
        label.move(50, 50)

        wave_type_label = QtWidgets.QLabel('Type:', self)
        wave_type_label.move(50, 100)
        self.wave_type_combo = QtWidgets.QComboBox(self)
        self.wave_type_combo.addItems(['sine', 'cosine'])
        self.wave_type_combo.setCurrentIndex(0)
        self.wave_type_combo.move(200, 100)

        amp_label = QtWidgets.QLabel('Amplitude:', self)
        amp_label.move(50, 150)
        self.amp_txt_box = QtWidgets.QLineEdit(self)
        self.amp_txt_box.move(200, 150)

        phase_shift_label = QtWidgets.QLabel('Phase Shift:', self)
        phase_shift_label.move(50, 200)
        self.phase_shift_txt_box = QtWidgets.QLineEdit(self)
        self.phase_shift_txt_box.move(200, 200)

        analog_freq_label = QtWidgets.QLabel('Analog Frequency:', self)
        analog_freq_label.move(50, 250)
        self.analog_freq_txt_box = QtWidgets.QLineEdit(self)
        self.analog_freq_txt_box.move(200, 250)

        sampling_freq_label = QtWidgets.QLabel('Sampling Frequency:', self)
        sampling_freq_label.move(50, 300)
        self.sampling_freq_txt_box = QtWidgets.QLineEdit(self)
        self.sampling_freq_txt_box.move(200, 300)

        test_signal_btn = QtWidgets.QPushButton('Test Signal', self)
        test_signal_btn.clicked.connect(self.test_signal)
        test_signal_btn.move(300, 100)

        # Button to generate discrete signal
        generate_discrete_signal_btn = QtWidgets.QPushButton('Generate Discrete Signal', self)
        generate_discrete_signal_btn.clicked.connect(self.generate_discrete_signal)
        generate_discrete_signal_btn.move(30, 350)

        # Button to generate continuous signal
        generate_continuous_signal_btn = QtWidgets.QPushButton('Generate Continuous Signal', self)
        generate_continuous_signal_btn.clicked.connect(self.generate_continuous_signal)
        generate_continuous_signal_btn.move(200, 350)

    def get_signal_data(self):
        self.selected_option = self.wave_type_combo.currentText()
        self.amp = float(self.amp_txt_box.text())
        self.phase_shift = float(self.phase_shift_txt_box.text())
        self.analog_freq = float(self.analog_freq_txt_box.text())
        self.sampling_freq = float(self.sampling_freq_txt_box.text())

    def generate_signal_Discrete(self):
        self.get_signal_data()
        if self.sampling_freq == 0:
            self.sampling_freq = 2 * self.analog_freq
            self.x_axis = np.arange(0, self.sampling_freq, 1)
            if self.selected_option == 'sine':
                self.y_axis = self.amp * np.sin(
                    2 * np.pi * (self.analog_freq / self.sampling_freq) * self.x_axis + self.phase_shift)
            else:
                self.y_axis = self.amp * np.cos(
                    2 * np.pi * (self.analog_freq / self.sampling_freq) * self.x_axis + self.phase_shift)

        elif self.sampling_freq < 2 * self.analog_freq:
            QtWidgets.QMessageBox.information(self, 'ERROR', "Invalid sampling_freq")
        else:
            self.x_axis = np.arange(0, self.sampling_freq, 1)
            if self.selected_option == 'sine':
                self.y_axis = self.amp * np.sin(
                    2 * np.pi * (self.analog_freq / self.sampling_freq) * self.x_axis + self.phase_shift)
            else:
                self.y_axis = self.amp * np.cos(
                    2 * np.pi * (self.analog_freq / self.sampling_freq) * self.x_axis + self.phase_shift)

    def generate_signal_Continuous(self):
        self.get_signal_data()
        if self.sampling_freq == 0:
            self.sampling_freq = 2 * self.analog_freq
            self.x_axis = np.arange(0, self.sampling_freq, 1)
            if self.selected_option == 'sine':
                self.y_axis = self.amp * np.sin(2 * np.pi * self.analog_freq * self.x_axis + self.phase_shift)
            else:
                self.y_axis = self.amp * np.cos(2 * np.pi * self.analog_freq * self.x_axis + self.phase_shift)
        elif self.sampling_freq < 2 * self.analog_freq:
            QtWidgets.QMessageBox.information(self, 'ERROR', "Invalid sampling_freq")
        else:
            self.x_axis = np.arange(0, self.sampling_freq, 1)
            if self.selected_option == 'sine':
                self.y_axis = self.amp * np.sin(2 * np.pi * self.analog_freq * self.x_axis + self.phase_shift)
            else:
                self.y_axis = self.amp * np.cos(2 * np.pi * self.analog_freq * self.x_axis + self.phase_shift)

    def test_signal(self):
        self.generate_signal_Discrete()

        if self.selected_option == 'sine':
            test_message = SignalSamplesAreEqual('SinOutput.txt', self.y_axis)
        else:
            test_message = SignalSamplesAreEqual('CosOutput.txt', self.y_axis)

        QtWidgets.QMessageBox.information(self, 'Test Result', test_message)

    def generate_discrete_signal(self):
        self.generate_signal_Discrete()
        plt.stem(self.x_axis, self.y_axis)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Generated Signal (Discrete)')
        plt.grid(True)
        plt.show()

    def generate_continuous_signal(self):
        self.generate_signal_Discrete()
        plt.plot(self.x_axis, self.y_axis)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Generated Signal (Continuous)')
        plt.grid(True)
        plt.show()
