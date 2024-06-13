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
import ECG
import FIR
import Fast
import DFT_IDFT
import TimeDomin
import GenerateTEST
import Opretions
import Display
import Quantization
import Correlation
import DCT


class DSP(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.login_window = None
        self.ECG_window = None
        self.FIR_window = None
        self.Fast_window = None
        self.Correlation_window = None
        self.TimeDomain_window = None
        self.Generate_Test_choice = None
        self.Generate_Test = None
        self.Choose_Display_choice = None
        self.DCT_window = None
        self.DFT_IDFT_window = None
        self.Quantization_window = None
        self.operations_window = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('DSP Tasks')
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(200, 0, self.width(), self.height())
        Choose_Display_btn = QtWidgets.QPushButton('Choose and Display', self)
        Choose_Display_btn.clicked.connect(self.open_Choose_Display_window)
        Choose_Display_btn.move(300, 350)
        Generate_Test_btn = QtWidgets.QPushButton('Generate and Test  ', self)
        Generate_Test_btn.clicked.connect(self.open_Generate_Test_window)
        Generate_Test_btn.move(150, 350)
        operations_btn = QtWidgets.QPushButton('Operations', self)
        operations_btn.clicked.connect(self.open_operations_window)
        operations_btn.move(450, 350)
        operations_btn = QtWidgets.QPushButton('Quantization', self)
        operations_btn.clicked.connect(self.open_Quantization_window)
        operations_btn.move(580, 350)
        operations_btn = QtWidgets.QPushButton('DFT/IDFT', self)
        operations_btn.clicked.connect(self.open_DFT_IDFT_window)
        operations_btn.move(150, 400)
        operations_btn = QtWidgets.QPushButton('DCT', self)
        operations_btn.clicked.connect(self.open_DCT_window)
        operations_btn.move(300, 400)
        operations_btn = QtWidgets.QPushButton('Time Domain', self)
        operations_btn.clicked.connect(self.open_TimeDomain_window)
        operations_btn.move(450, 400)
        operations_btn = QtWidgets.QPushButton('Correlation', self)
        operations_btn.clicked.connect(self.open_Correlation_window)
        operations_btn.move(580, 400)
        operations_btn = QtWidgets.QPushButton('Fast', self)
        operations_btn.clicked.connect(self.open_Fast_window)
        operations_btn.move(150, 450)
        operations_btn = QtWidgets.QPushButton('FIR', self)
        operations_btn.clicked.connect(self.open_FIR_window)
        operations_btn.move(300, 450)
        operations_btn = QtWidgets.QPushButton('ECG', self)
        operations_btn.clicked.connect(self.open_ECG_window)
        operations_btn.move(450, 450)
        operations_btn = QtWidgets.QPushButton('End', self)
        operations_btn.clicked.connect(self.open_login_window)
        operations_btn.move(580, 450)

    def open_Choose_Display_window(self):
        self.Choose_Display_choice = Display.Choose_Display()
        self.Choose_Display_choice.show()

    def open_Generate_Test_window(self):
        self.Generate_Test_choice = GenerateTEST.Generate_Test()
        self.Generate_Test_choice.show()

    def open_operations_window(self):
        self.operations_window = Opretions.Operations()
        self.operations_window.show()

    def open_Quantization_window(self):
        self.Quantization_window = Quantization.Quantization()
        self.Quantization_window.show()

    def open_DFT_IDFT_window(self):
        self.DFT_IDFT_window = DFT_IDFT.DFT_IDFT()
        self.DFT_IDFT_window.show()

    def open_DCT_window(self):
        self.DCT_window = DCT.DCT()
        self.DCT_window.show()

    def open_TimeDomain_window(self):
        self.TimeDomain_window = TimeDomin.TimeDomain()
        self.TimeDomain_window.show()

    def open_Correlation_window(self):
        self.Correlation_window = Correlation.Correlation()
        self.Correlation_window.show()

    def open_Fast_window(self):
        self.Fast_window = Fast.Fast()
        self.Fast_window.show()

    def open_FIR_window(self):
        self.FIR_window = FIR.FIR()
        self.FIR_window.show()

    def open_ECG_window(self):
        self.ECG_window = ECG.ECG()
        self.ECG_window.show()

    def open_login_window(self):
        self.login_window = MainFun()
        self.login_window.show()


class MainFun(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.DSP = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Welcome to tasks')
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(200, 0, self.width(), self.height())
        DSP_Button = QtWidgets.QPushButton('DSP TASKS ', self)
        DSP_Button.clicked.connect(self.open_DSP_Program)
        DSP_Button.move(350, 350)

    def open_DSP_Program(self):
        self.DSP = DSP()
        self.DSP.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainfun = MainFun()
    mainfun.show()
    sys.exit(app.exec_())
