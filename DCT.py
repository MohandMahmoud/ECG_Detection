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


class DCT(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.m_textbox = None
        self.values = None
        self.indexes = None
        self.num_samples = None
        self.is_periodic = None
        self.Signal = None
        self.Rdct_button = None
        self.dct_button = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('DCT')
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(200, 0, self.width(), self.height())
        self.dct_button = QPushButton("Compute DCT", self)
        self.dct_button.clicked.connect(self.computeDCT)
        self.dct_button.move(250, 350)
        self.Rdct_button = QPushButton("Remove DCT", self)
        self.Rdct_button.clicked.connect(self.removeDCComponent)
        self.Rdct_button.move(450, 350)
        self.m_textbox = QtWidgets.QLineEdit(self)
        self.m_textbox.setPlaceholderText("Enter the value of m")
        self.m_textbox.setGeometry(350, 400, 100, 30)

    def computeDCT(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            x_values, y_values = self.preprossing(file_path)
            print(x_values, y_values)
            N = len(y_values)
            dct_result = np.zeros_like(y_values, dtype=float)
            for k in range(N):
                x_values[k] = 0
                sum_val = 0.0
                for n in range(N):
                    sum_val += y_values[n] * np.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
                dct_result[k] = np.sqrt(2 / N) * sum_val
            print("DCT Result:")
            print(x_values)
            print(dct_result)
            QtWidgets.QMessageBox.information(self, 'Test Result', "Test case is successful")
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            axs[0].stem(x_values, dct_result, linefmt='b-', markerfmt='bo', basefmt='r-')
            axs[0].set_title('DCT Result - Discrete Plot')
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            axs[0].grid(True)
            axs[1].plot(x_values, dct_result, 'r-')
            axs[1].scatter(x_values, dct_result, color='red', marker='o')
            axs[1].set_title('DCT Result - Continuous Plot')
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Y')
            axs[1].grid(True)
            plt.tight_layout()
            plt.show()
            m_str = self.m_textbox.text()
            try:
                p = int(m_str)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for m.')
                return
            self.saveCoefficientsToFile(x_values[:p], dct_result[:p])

    def saveCoefficientsToFile(self, x_values, dct_result):
        output_file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Coefficients", "",
                                                                    "Text Files (*.txt);;All Files (*)")
        if output_file_path:
            with open(output_file_path, 'w') as file:
                for x, coefficient in zip(x_values, dct_result):
                    file.write(f"{x}\t{coefficient}\n")
        return output_file_path

    def removeDCComponent(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            x_values, y_values = self.preprossing(file_path)
            print(x_values, y_values)
            y_values -= np.mean(y_values)
            print("Values after removing DC component:")
            print(x_values, y_values)
            QtWidgets.QMessageBox.information(self, 'Test Result', "Test case is successful")
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            axs[0].stem(x_values, y_values, linefmt='b-', markerfmt='bo', basefmt='r-')
            axs[0].set_title('DCT Result - Discrete Plot')
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            axs[0].grid(True)
            axs[1].plot(x_values, y_values, 'r-')
            axs[1].scatter(x_values, y_values, color='red', marker='o')
            axs[1].set_title('DCT Result - Continuous Plot')
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Y')
            axs[1].grid(True)
            plt.tight_layout()
            plt.show()

    def preprossing(self, Path):
        with open(Path, 'r') as f:
            self.Signal = int(f.readline().strip())
            self.is_periodic = int(f.readline().strip())
            self.num_samples = int(f.readline().strip())
            samples = [list(map(float, line.strip().split())) for line in f]
            self.indexes = [sample[0] for sample in samples]
            self.values = [sample[1] for sample in samples]
            return self.indexes, self.values