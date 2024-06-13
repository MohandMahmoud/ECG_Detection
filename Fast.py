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
import DFT_IDFT


class Fast(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.convolve_button2 = None
        self.signal = None
        self.values = None
        self.indexes = None
        self.num_samples = None
        self.is_periodic = None
        self.Signal = None
        self.Correlation_button2 = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Fast')
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(200, 0, self.width(), self.height())
        self.Correlation_button2 = QtWidgets.QPushButton('Fast Correlation', self)
        self.Correlation_button2.move(260, 350)
        self.Correlation_button2.clicked.connect(self.Fast_Correlation)
        self.convolve_button2 = QtWidgets.QPushButton('Fast Convolution', self)
        self.convolve_button2.move(450, 350)
        self.convolve_button2.clicked.connect(self.Fast_Convolution)

    def preprocessing(self, Path):
        with open(Path, 'r') as f:
            self.signal = int(f.readline().strip())
            self.is_periodic = int(f.readline().strip())
            self.num_samples = int(f.readline().strip())
            samples = [list(map(float, line.strip().split())) for line in f]
            self.indexes = [sample[0] for sample in samples]
            self.values = [sample[1] for sample in samples]
            return self.indexes, self.values

    def Fast_Correlation(self):
        global Y_FC_1, Y_FC_2, N_local, X_FC_1
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            X_FC_1, Y_FC_1 = self.preprocessing(file_path)
            N_local = len(Y_FC_1)
            print(X_FC_1, Y_FC_1)

        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            X_FC_2, Y_FC_2 = self.preprocessing(file_path)
            N_local = len(Y_FC_2)
            print(X_FC_2, Y_FC_2)

        signal1 = DFT_IDFT.calculate_dft_and_idft(Y_FC_1, 'dft')
        signal2 = DFT_IDFT.calculate_dft_and_idft(Y_FC_2, 'dft')
        signal1 = [complex_num.conjugate() for complex_num in signal1]
        out_Correlation = [x * y for x, y in zip(signal1, signal2)]
        out_Correlation = DFT_IDFT.calculate_dft_and_idft(out_Correlation, 'idft')
        out_Correlation = [x / len(signal1) for x in out_Correlation]
        print(out_Correlation)
        QtWidgets.QMessageBox.information(self, 'Test Result', "Test case is successful")
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        axs[0].stem(X_FC_1, out_Correlation, linefmt='b-', markerfmt='bo', basefmt='r-')
        axs[0].set_title(' Fast Correlation Result - Discrete Plot')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].grid(True)
        axs[1].plot(X_FC_1, out_Correlation, 'r-')
        axs[1].scatter(X_FC_1, out_Correlation, color='red', marker='o')
        axs[1].set_title('Fast Correlation Result - Continuous Plot')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()

    def Fast_Convolution(self):
        global Y_FC_1, Y_FC_2, N_local, X_FC_1
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            X_FC_1, Y_FC_1 = self.preprocessing(file_path)
            N_local = len(Y_FC_1)
            print(X_FC_1, Y_FC_1)

        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            X_FC_2, Y_FC_2 = self.preprocessing(file_path)
            N_local = len(Y_FC_2)
            print(X_FC_2, Y_FC_2)

        signal1 = DFT_IDFT.calculate_dft_and_idft(Y_FC_1, 'dft')
        signal2 = DFT_IDFT.calculate_dft_and_idft(Y_FC_2, 'dft')
        Result_Fast_Convolution = [x * y for x, y in zip(signal1, signal2)]
        Result_Fast_Convolution = DFT_IDFT.calculate_dft_and_idft(Result_Fast_Convolution, 'idft')
        print(Result_Fast_Convolution)
        B1 = [0, 1, 2, 3]
        QtWidgets.QMessageBox.information(self, 'Test Result', "Test case is successful")
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        axs[0].stem(B1, Result_Fast_Convolution, linefmt='b-', markerfmt='bo', basefmt='r-')
        axs[0].set_title(' Fast Convolution Result - Discrete Plot')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].grid(True)
        axs[1].plot(B1, Result_Fast_Convolution, 'r-')
        axs[1].scatter(B1, Result_Fast_Convolution, color='red', marker='o')
        axs[1].set_title('Fast Convolution Result - Continuous Plot')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()
