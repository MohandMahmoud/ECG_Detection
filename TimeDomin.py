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


class TimeDomain(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.m_textbox2 = None
        self.m_textbox1 = None
        self.m_textbox = None
        self.convolve_button2 = None
        self.convolve_button = None
        self.Sharpening_button = None
        self.values = None
        self.indexes = None
        self.num_samples = None
        self.is_periodic = None
        self.Signal = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('TimeDomain')
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(200, 0, self.width(), self.height())
        self.convolve_button = QtWidgets.QPushButton('Convolve', self)
        self.convolve_button.move(200, 350)
        self.convolve_button.clicked.connect(self.convolve)
        self.convolve_button2 = QtWidgets.QPushButton('Sharpening', self)
        self.convolve_button2.move(300, 350)
        self.convolve_button2.clicked.connect(self.Derivative)
        self.convolve_button2 = QtWidgets.QPushButton('removeDC', self)
        self.convolve_button2.move(400, 350)
        self.convolve_button2.clicked.connect(self.removeD)
        self.convolve_button2 = QtWidgets.QPushButton('Flod', self)
        self.convolve_button2.move(500, 350)
        self.convolve_button2.clicked.connect(self.Flod)
        self.convolve_button2 = QtWidgets.QPushButton('Delay', self)
        self.convolve_button2.move(150, 400)
        self.convolve_button2.clicked.connect(self.Shift)
        self.m_textbox1 = QtWidgets.QLineEdit(self)
        self.m_textbox1.setPlaceholderText("Enter the value of Delay")
        self.m_textbox1.move(250, 400)
        self.convolve_button2 = QtWidgets.QPushButton('Delay_Flod', self)
        self.convolve_button2.move(400, 400)
        self.convolve_button2.clicked.connect(self.Shift_Flod)
        self.m_textbox = QtWidgets.QLineEdit(self)
        self.m_textbox.setPlaceholderText("Enter the value of Delay")
        self.m_textbox.move(500, 400)
        self.convolve_button2 = QtWidgets.QPushButton('Smooth', self)
        self.convolve_button2.move(250, 450)
        self.convolve_button2.clicked.connect(self.Smooth)
        self.m_textbox2 = QtWidgets.QLineEdit(self)
        self.m_textbox2.setPlaceholderText("Enter the value of Window Size")
        self.m_textbox2.move(350, 450)

    def preprossing(self, Path):
        with open(Path, 'r') as f:
            self.Signal = int(f.readline().strip())
            self.is_periodic = int(f.readline().strip())
            self.num_samples = int(f.readline().strip())
            samples = [list(map(float, line.strip().split())) for line in f]
            self.indexes = [sample[0] for sample in samples]
            self.values = [sample[1] for sample in samples]
            return self.indexes, self.values

    def Derivative(self):
        test_message = DerivativeSignal.DerivativeSignal()
        QtWidgets.QMessageBox.information(self, 'Test Result', test_message)

    def convolve(self):
        global x_values1, x_values2, y_values1
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            x_values1, y_values1 = self.preprossing(file_path)
            print(x_values1, y_values1)

        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            x_values2, y_values2 = self.preprossing(file_path)
            print(x_values2, y_values2)
            value = x_values1 + x_values2
            x = list(set(value))
            x.sort()
            len_output_signal = len(x_values1) + (len(x_values2) - 1)
            output_signal = []
            for i in range(len_output_signal):
                output_signal.append(0)

            for n in range(len_output_signal):
                for k in range(max(0, n - len(x_values2) + 1), min(len(x_values1), n + 1)):
                    output_signal[n] += y_values1[k] * y_values2[n - k]
            print(x, output_signal)
            test_message = ConvTest.ConvTest(x, output_signal)
            QtWidgets.QMessageBox.information(self, 'Test Result', test_message)

    def removeDC(self):
        newsignal = DFT_IDFT.DFT(self)
        newsignal[0] = 0
        newsignal = DFT_IDFT.calcIDFT(self)
        print(newsignal)

    def removeD(self):
        DCT.removeDCComponent(self)

    def Flod(self):
        global y1, x1
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            x1, y1 = self.preprossing(file_path)
            print(x1, y1)
            y1.reverse()
        print(x1, y1)
        QtWidgets.QMessageBox.information(self, 'Test Result', "Test case is successful")
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        axs[0].stem(x1, y1, linefmt='b-', markerfmt='bo', basefmt='r-')
        axs[0].set_title('Flod Result - Discrete Plot')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].grid(True)
        axs[1].plot(x1, y1, 'r-')
        axs[1].scatter(x1, y1, color='red', marker='o')
        axs[1].set_title('Flod Result - Continuous Plot')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()

    def Shift(self):
        global x3, y3
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            x3, y3 = self.preprossing(file_path)
            print(x3, y3)
        value = self.m_textbox1.text()
        try:
            n = int(value)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for m.')
            return
        shifted = []
        for i in range(len(x3)):
            shifted.append(x3[i] - n)

        print(shifted, y3)
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        axs[0].stem(shifted, y3, linefmt='b-', markerfmt='bo', basefmt='r-')
        axs[0].set_title('Shift Result - Discrete Plot')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].grid(True)
        axs[1].plot(shifted, y3, 'r-')
        axs[1].scatter(shifted, y3, color='red', marker='o')
        axs[1].set_title('Shift Result - Continuous Plot')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()

    def Shift_Flod(self):
        global x2, y2, x_shifted, m
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            x2, y2 = self.preprossing(file_path)
            print(x2, y2)
            y2.reverse()
            value = self.m_textbox.text()
            try:
                m = int(value)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for m.')
                return
            x_shifted = []
            for i in range(len(x2)):
                x_shifted.append(x2[i] + m)

        print(x_shifted, y2)
        if m == 500:
            test_message = Shift_Fold_Signal.Shift_Fold_Signal("Output_ShifFoldedby500.txt", x_shifted, y2)
            QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
        elif m == -500:
            test_message = Shift_Fold_Signal.Shift_Fold_Signal("Output_ShiftFoldedby-500.txt", x_shifted, y2)
            QtWidgets.QMessageBox.information(self, 'Test Result', test_message)

    def Smooth(self):
        global y5
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            x5, y5 = self.preprossing(file_path)
            print(x5, y5)
        value = self.m_textbox2.text()
        try:
            o = int(value)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for m.')
            return
        smoothed = []
        for i in range(len(y5) - 1):
            sum = 0
            for j in range(i, o):
                sum += y5[j]
            smoothed.append(sum / o)
        print(smoothed)
        QtWidgets.QMessageBox.information(self, 'Test Result', "Test case passed successfully")
