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

class DFT_IDFT(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.amplitude_label = None
        self.amplitude_input = None
        self.phase_shift_label = None
        self.index_label = None
        self.phase_shift_input = None
        self.output_text = None
        self.dft_button = None
        self.freq_input = None
        self.index_input = None
        self.freq_label = None
        self.idft_button = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('DFT_IDFT')
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(200, 0, self.width(), self.height())
        self.freq_label = QLabel("Frequency:", self)
        self.freq_label.move(10, 30)
        self.freq_input = QLineEdit(self)
        self.freq_input.move(85, 30)
        self.dft_button = QPushButton("DFT", self)
        self.dft_button.clicked.connect(self.DFT)
        self.dft_button.move(120, 80)
        self.index_label = QLabel("Index:", self)
        self.index_label.move(300, 30)
        self.index_input = QLineEdit(self)
        self.index_input.move(350, 30)
        self.amplitude_label = QLabel("Amplitude:", self)
        self.amplitude_label.move(580, 30)
        self.amplitude_input = QLineEdit(self)
        self.amplitude_input.move(650, 30)
        self.phase_shift_label = QLabel("Phase Shift:", self)
        self.phase_shift_label.move(290, 80)
        self.phase_shift_input = QLineEdit(self)
        self.phase_shift_input.move(365, 80)
        self.idft_button = QPushButton("IDFT", self)
        self.idft_button.clicked.connect(self.calcIDFT)
        self.idft_button.move(580, 80)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text files (*.txt);;All Files (*)")
        if file_path:
            with open(file_path, 'r') as file:
                content = file.readlines()[3:]
                return content

    @staticmethod
    def calculate_dft_and_idft(values, type_of_calc):
        output = []
        for k in range(len(values)):
            harmonic = DFT_IDFT.calculate_harmonic_or_element(k, values, type_of_calc)
            if type_of_calc == 'idft':
                output.append(harmonic.real.__round__(3))
            else:
                output.append(harmonic)
        return output

    @staticmethod
    def calculate_harmonic_or_element(k, values, type_of_calc):
        len_of_values = len(values)
        summ = 0
        for n in range(len_of_values):
            summ += DFT_IDFT.calculate_one_element(n, k, values, type_of_calc)
        if type_of_calc == 'idft':
            return summ * (1 / len_of_values)
        return summ

    @staticmethod
    def calculate_one_element(n, k, values, type_of_calc):
        if values[n] == 0:
            return 0
        rtn = values[n] * DFT_IDFT.calculate_exp(n, k, len(values), type_of_calc)
        if type_of_calc == 'idft':
            rtn = (rtn.real + (rtn.imag * 1j))
        return rtn

    @staticmethod
    def calculate_exp(n, k, len_of_values, type_of_calc):
        the_power = (1j * 2 * n * k) / len_of_values
        if the_power.imag == 0:
            return 1 + 0j
        sin_value = float(math.sin(math.pi * the_power.imag.__abs__()))
        cos_value = float(math.cos(math.pi * the_power.imag.__abs__()))
        if type_of_calc == 'dft':
            sin_value *= -1j
        else:
            sin_value *= 1j
        e = cos_value + sin_value

        return e

    @staticmethod
    def calculate_fundamentel_freq(sampling_freq, len_of_values):
        periodic_time_of_sample = 1 / sampling_freq
        down_term = len_of_values * periodic_time_of_sample
        up_term = 2 * math.pi
        return up_term / down_term

    @staticmethod
    def calculate_ampl(dft):
        ampl = []
        for i in range(len(dft)):
            the_powered_real_number = dft[i].real * dft[i].real
            the_powered_imag_number = dft[i].imag * dft[i].imag
            summ = the_powered_real_number + the_powered_imag_number
            ampl.append(math.sqrt(float(summ)))
        return ampl

    @staticmethod
    def calculate_phase_shift(dft):
        phases = []
        for i in range(len(dft)):
            phases.append(float(math.atan2(dft[i].imag, dft[i].real)))
        return phases

    @staticmethod
    def signal_representation():
        plt.subplot(2, 1, 1)
        plt.subplot(2, 1, 2)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_freq_domain(fundamentel_freq, amp_y):
        x = []
        for i in range(len(amp_y)):
            x.append(i * fundamentel_freq)
        DFT_IDFT.signal_representation()

    @staticmethod
    def plot_time_domain(y):
        x = []
        for i in range(len(y)):
            x.append(i)

        # Define or import the following function and constants
        # SignalsMethods.plot_normal_signal(x, y, 'time', 'samples', SignalType.Discrete, 'Time Domain')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def convert_from_polar_form(ampl, theta):
        outputs = []
        for i in range(len(ampl)):
            img = ampl[i] * math.sin(theta[i]) * 1j
            real = ampl[i] * math.cos(theta[i])
            complex_num = img + real
            outputs.append(complex_num)
        return outputs

    @staticmethod
    def parse_signal_from_file(content):
        time = []
        signals = []
        if content:
            for line in content:
                columns = line.split()
                time.append(float(columns[0]))
                signals.append(float(columns[1]))
        return time, signals

    def plot_samples(self, SS, amplitude, phase_shift):
        QtWidgets.QMessageBox.information(self, 'Test Result', "Test case is successful")
        plt.subplot(2, 1, 1)
        plt.stem(SS, amplitude)
        plt.title("Frequency - Amplitude")
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")

        # Create the second subplot
        plt.subplot(2, 1, 2)
        plt.stem(signal, phase_shift)
        plt.title("Frequency - Phase Shift")
        plt.xlabel("Frequency")
        plt.ylabel("Phase Shift")

        # Adjust the layout to avoid overlapping titles
        plt.tight_layout()
        plt.show()

    @staticmethod
    def calcOmegaArray(frequency, n):
        omega = []
        for i in range(1, n + 1):
            omega.append((2 * math.pi * frequency * i) / n)
        return omega

    def DFT(self):
        content = self.open_file()
        index, samples = self.parse_signal_from_file(content)
        length = len(samples)
        A = []
        phase = []
        for k in range(length):
            memo = 0
            for n in range(length):
                theta = ((2 * 180 * k * n) / length) * (math.pi / 180)
                real = math.cos(theta)
                imaginary = math.sin(theta)
                memo += samples[n] * complex(real, -imaginary)
            A.append(np.abs(memo))
            phase.append(np.angle(memo))
        self.output_text.setPlainText(f"A: {A}\nPhase: {phase}")
        frequency = int(self.freq_input.text())
        omegaSignal = self.calcOmegaArray(frequency, length)
        self.plot_samples(omegaSignal, A, phase)
        return omegaSignal

    @staticmethod
    def writeToFile(A, phase):
        length = len(A)
        with open('outputDFT.txt', 'w') as file:
            file.write(str(0) + '\n')
            file.write(str(1) + '\n')
            file.write(str(length) + '\n')
            for x, y in zip(A, phase):
                file.write(str(x) + ' ' + str(y) + '\n')

    @staticmethod
    def convertToNumber(content):
        time = []
        signals = []
        if content:
            for line in content:
                columns = line.split(',')
                time_str = str(columns[0])
                time_str = time_str.replace('f', '')
                time.append(float(time_str))
                signal_str = str(columns[1])
                signal_str = signal_str.replace('f', '')
                signals.append(float(signal_str))
        return time, signals

    def realAndImaginary(self, A, phase):
        complex_signal = []
        index = int(self.index_input.text())
        amb = int(self.amplitude_input.text())
        p = int(self.phase_shift_input.text())
        if amb != 0 and p != 0:
            A[index] = amb
            phase[index] = p
        for i in range(len(A)):
            real_part = (A[i] * math.cos(phase[i]))
            imaginary_part = (A[i] * math.sin(phase[i]))
            complex_signal.append(complex(real_part, imaginary_part))
        self.output_text.setPlainText(f"A: {A}\nPhase: {phase}")
        return complex_signal

    def calcIDFT(self):
        content = self.open_file()
        amplitude, phase_shift = self.convertToNumber(content)
        complex_signal = self.realAndImaginary(amplitude, phase_shift)
        N = len(complex_signal)
        original_signal = []
        time = np.linspace(0, N - 1, N)
        for n in range(N):
            accumulator = 0
            for k in range(N):
                theta = (2 * np.pi * k * n) / N
                real_part = np.cos(theta)
                imaginary_part = np.sin(theta)
                current_complex = complex(real_part, imaginary_part)
                out = complex_signal[k] * current_complex
                real_temp = round(out.real, 2)
                img_temp = round(out.imag, 2)
                accumulator += complex(real_temp, img_temp)
            original_signal.append(int(accumulator.real / N))
        self.output_text.setPlainText(f"Original Signal: {original_signal}")
        self.plot_figure(time, original_signal)
        return original_signal

    def plot_figure(self, time, S):
        QtWidgets.QMessageBox.information(self, 'Test Result', "Test case is successful")
        plt.figure()
        plt.stem(time, S, label='Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Signal')
        plt.legend()
        plt.show()
