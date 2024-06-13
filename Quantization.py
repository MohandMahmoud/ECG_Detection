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


class Quantization(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantization")
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(200, 0, self.width(), self.height())
        self.button_task1 = QtWidgets.QPushButton("Test 1", self)
        self.button_task1.move(250, 350)
        self.button_task1.clicked.connect(self.Test1)
        self.button_task2 = QtWidgets.QPushButton("Test 2", self)
        self.button_task2.move(450, 350)
        self.button_task2.clicked.connect(self.Test2)

    def Test1(self):
        num = self.get_input_dialog("Enter number of bits")
        if num is not None:
            x, y = self.get_data_file("C:/Users\Lenovo\PycharmProjects\Honda\Quan1_input.txt")
            number = 2 ** num
            Min = min(y)
            Max = max(y)
            delta = (Max - Min) / number
            q = []
            ra = []
            tem = Min
            mmm = []
            nnn = []
            b = num
            n = number
            for i in range(int(number)):
                temp = round(float(tem + delta), 4)
                mid = round((temp + tem) / 2, 4)
                q.append(mid)
                nnn.append(tem)
                tem = temp
                mmm.append(temp)
            for i in range(len(y)):
                for x in range(n):
                    if nnn[x] <= y[i] <= mmm[x]:
                        ra.append(x)
                        break
            results = []
            for i in range(len(ra)):
                results.append(q[ra[i]])
            encod = []
            bin3 = lambda z: ''.join(reversed([str((z >> c) & 1) for c in range(b)]))
            for i in range(len(ra)):
                encod.append(bin3(ra[i]))
            test_message = QuantizationTest1("Quan1_Out.txt", encod, results)
            QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
            plt.stem(encod, results)
            plt.plot(encod, results)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title('Generated Signal (Continuous)')
            plt.grid(True)
            plt.show()

    @staticmethod
    def get_data_file(path):
        with open(path) as f:
            data = f.readlines()
        x_values = []
        y_values = []
        for item in data:
            item = item.strip()
            if ' ' in item:
                x_v, y_v = item.split(' ')
                x_values.append(float(x_v))
                y_values.append(float(y_v))
        return x_values, y_values

    def Test2(self):
        num = self.get_input_dialog("Enter number of Levels")
        if num is not None:
            x, y = self.get_data_file("C:/Users\Lenovo\PycharmProjects\Honda\Quan2_input.txt")
            numb = math.log(num, 2)
            Min = min(y)
            Max = max(y)
            delta = (Max - Min) / num
            delta = round(delta, 3)
            q = []
            ra = []
            tem = Min
            mmm = []
            nnn = []
            n = int(numb)
            for i in range(len(y)):
                temp = round(float(tem + delta), 3)
                mid = round((temp + tem) / 2, 3)
                q.append(mid)
                nnn.append(tem)
                tem = temp
                mmm.append(temp)
            for i in range(len(y)):
                for x in range(len(y)):
                    if mmm[x] >= y[i] >= nnn[x]:
                        ra.append(x)
                        break
            results = []
            for i in range(len(ra)):
                results.append(q[ra[i]])
            encod = []
            bin3 = lambda t: ''.join(reversed([str((t >> b) & 1) for b in range(n)]))
            for i in range(len(ra)):
                encod.append(bin3(ra[i]))
            er = []
            for i in range(len(ra)):
                a = float(results[i] - y[i])
                a = round(a, 3)
                er.append(a)
            for i in range(len(ra)):
                ra[i] = ra[i] + 1
            test_message = QuantizationTest2("Quan2_Out.txt", ra, encod, results, er)
            QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
            plt.stem(encod, results)
            plt.plot(encod, results)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title('Generated Signal (Continuous)')
            plt.grid(True)
            plt.show()

    def get_input_dialog(self, prompt):
        num, ok = QtWidgets.QInputDialog.getInt(self, "Input", prompt, 0, 1, 100)
        if ok:
            return num
        return None
