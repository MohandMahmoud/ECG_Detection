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


class Choose_Display(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.choose1_btn = None
        self.choose2_btn = None
        self.signal_one = None
        self.signal_two = None
        self.display_btn = None
        self.signal_one_type = None
        self.is_periodic_one = None
        self.num_samples_one = None
        self.indexes_one = None
        self.values_one = None
        self.signal_two_type = None
        self.is_periodic_two = None
        self.num_samples_two = None
        self.indexes_two = None
        self.values_two = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Choose and Display Signals')
        self.setGeometry(100, 100, 800, 500)

        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)

        image_label.setGeometry(200, 0, self.width(), self.height())

        self.choose1_btn = QtWidgets.QPushButton('Choose Signal 1', self)
        self.choose1_btn.clicked.connect(self.choose_signal_one)
        self.choose1_btn.move(300, 350)

        self.choose2_btn = QtWidgets.QPushButton('Choose Signal 2', self)
        self.choose2_btn.clicked.connect(self.choose_signal_two)
        self.choose2_btn.move(450, 350)

        self.display_btn = QtWidgets.QPushButton('Display Signal', self)
        self.display_btn.clicked.connect(self.display_signal)
        self.display_btn.move(380, 400)

    def choose_signal_one(self):
        file_dialog = QFileDialog()
        self.signal_one, _ = file_dialog.getOpenFileName(self, 'Choose First Signal', '', 'Text Files (*.txt)')
        if self.signal_one:
            with open(self.signal_one, 'r') as f:
                self.signal_one_type = int(f.readline().strip())
                self.is_periodic_one = int(f.readline().strip())
                self.num_samples_one = int(f.readline().strip())
                samples_one = [list(map(float, line.strip().split())) for line in f]
                self.indexes_one = [sample[0] for sample in samples_one]
                self.values_one = [sample[1] for sample in samples_one]

    def choose_signal_two(self):
        file_dialog = QFileDialog()
        self.signal_two, _ = file_dialog.getOpenFileName(self, 'Choose Second Signal', '', 'Text Files (*.txt)')
        if self.signal_two:
            with open(self.signal_two, 'r') as f:
                self.signal_two_type = int(f.readline().strip())
                self.is_periodic_two = int(f.readline().strip())
                self.num_samples_two = int(f.readline().strip())
                samples_two = [list(map(float, line.strip().split())) for line in f]
                self.indexes_two = [sample[0] for sample in samples_two]
                self.values_two = [sample[1] for sample in samples_two]

    def display_signal(self):
        if self.signal_one_type == 0:
            plt.subplot(2, 2, 1)
            plt.plot(self.indexes_one, self.values_one)
            plt.scatter(self.indexes_one, self.values_one)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title('Continuous Signal')
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.stem(self.indexes_one, self.values_one, basefmt=' ', use_line_collection=True)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title('Discrete Signal')
            plt.grid(True)

        if self.signal_two:
            if self.signal_two_type == 0:
                plt.subplot(2, 2, 2)
                plt.plot(self.indexes_two, self.values_two)
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.title('Continuous Signal')
                plt.grid(True)

                plt.subplot(2, 2, 4)
                plt.stem(self.indexes_two, self.values_two, basefmt=' ', use_line_collection=True)
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.title('Discrete Signal')
                plt.grid(True)

        plt.tight_layout()
        plt.show()
