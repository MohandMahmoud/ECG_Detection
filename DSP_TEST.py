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


class DCTTransform:
    @staticmethod
    def calculate_angle(values_len, n, k):
        element_one = math.pi / (4 * values_len)
        element_two = (2 * n) - 1
        element_three = (2 * k) - 1
        result = element_one * element_two * element_three
        return math.cos(result)

    @staticmethod
    def calculate_one_element(signal_values, n, k):
        angle = DCTTransform.calculate_angle(len(signal_values), n, k)
        return signal_values[n] * angle

    @staticmethod
    def calculate_sum(signal_values, k):
        summ = 0
        for n in range(len(signal_values)):
            summ += DCTTransform.calculate_one_element(signal_values, n, k)
        return summ

    @staticmethod
    def dct_transform(signal_values):
        y_values = []
        values_len = len(signal_values)
        value_under_root = 2 / values_len
        for k in range(values_len):
            result = math.sqrt(value_under_root) * DCTTransform.calculate_sum(signal_values, k)
            y_values.append(result)
        return y_values

    @staticmethod
    def calculate_mean_of_signal(signal_values):
        summ = 0
        len_of_values = len(signal_values)
        for i in range(len_of_values):
            summ += signal_values[i]
        return summ / len_of_values

    @staticmethod
    def remove_dc_component(signal_values):
        len_of_values = len(signal_values)
        removed_values = []
        for i in range(len_of_values):
            result = signal_values[i] - DCTTransform.calculate_mean_of_signal(signal_values)
            removed_values.append(round(result, 3))
        return removed_values