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
import DSP_TEST
class ECG(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.NewFS_textbox1 = None
        self.FS_textbox1 = None
        self.maxF_textbox1 = None
        self.miniF_textbox1 = None
        self.Preprossing_ECG_button = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ECG')
        self.setGeometry(100, 100, 800, 500)
        image_label = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap("images.png")
        image_label.setPixmap(pixmap)
        image_label.setGeometry(200, 0, self.width(), self.height())
        self.Preprossing_ECG_button = QtWidgets.QPushButton('ECG', self)
        self.Preprossing_ECG_button.move(200, 350)
        self.Preprossing_ECG_button.clicked.connect(self.Preprossing_ECG)
        self.miniF_textbox1 = QtWidgets.QLineEdit(self)
        self.miniF_textbox1.setPlaceholderText("Enter miniF")
        self.miniF_textbox1.move(300, 350)
        self.maxF_textbox1 = QtWidgets.QLineEdit(self)
        self.maxF_textbox1.setPlaceholderText("Enter maxF")
        self.maxF_textbox1.move(450, 350)
        self.FS_textbox1 = QtWidgets.QLineEdit(self)
        self.FS_textbox1.setPlaceholderText("Enter FS")
        self.FS_textbox1.move(600, 350)
        self.NewFS_textbox1 = QtWidgets.QLineEdit(self)
        self.NewFS_textbox1.setPlaceholderText("Enter New FS")
        self.NewFS_textbox1.move(450, 400)

    @staticmethod
    def next_odd_num(num=0.0):
        if num.__ceil__() % 2 == 0:
            return num.__ceil__() + 1
        elif num.__ceil__() % 1 == 0:
            return num.__ceil__()

    def Preprossing_ECG(self):
        A_Folder = 'C:/Users/Lenovo/Downloads/Practical task 2/Practical task 2/A'
        A_signals = self.load_signals_from_folder(A_Folder)
        B_Folder = 'C:/Users/Lenovo\Downloads\Practical task 2/Practical task 2/B'
        B_signals = self.load_signals_from_folder(B_Folder)
        Test_Folder = 'C:/Users/Lenovo/Downloads/Practical task 2/Practical task 2/Test Folder'
        Test_signals = self.load_signals_from_folder(Test_Folder)
        num_signals = len(A_signals)
        fig, axs = plt.subplots(num_signals, 1, figsize=(8, 8 * num_signals))

        for idx, S_A in enumerate(A_signals):
            A_index = [i for i in range(len(S_A))]
            axs[idx].plot(A_index, S_A, 'r-')
            axs[idx].scatter(A_index, S_A, color='red', marker='o')
            axs[idx].set_title(f'A signal {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()
        num_signals_B = len(B_signals)
        fig, axs_B = plt.subplots(num_signals_B, 1, figsize=(8, 8 * num_signals_B))

        for idx, S_B in enumerate(B_signals):
            B_index = [i for i in range(len(S_B))]
            axs_B[idx].plot(B_index, S_B, 'b-')
            axs_B[idx].scatter(B_index, S_B, color='blue', marker='o')
            axs_B[idx].set_title(f'B signal {idx + 1} - Continuous Plot')
            axs_B[idx].set_xlabel('X')
            axs_B[idx].set_ylabel('Y')
            axs_B[idx].grid(True)
        plt.tight_layout()
        plt.show()

        num_signals_T = len(Test_signals)
        fig, axs = plt.subplots(num_signals_T, 1, figsize=(8, 8 * num_signals_T))

        for idx, S_T in enumerate(Test_signals):
            T_index = [i for i in range(len(S_T))]
            axs[idx].plot(T_index, S_T, 'r-')
            axs[idx].scatter(T_index, S_T, color='green', marker='o')
            axs[idx].set_title(f'Test signal {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()
        O_C=[]
        A_P=[]
        D_A=[]
        for S_A_F in A_signals:
            A_index = [i for i in range(len(S_A_F))]
            S_A_F = self.fillter(S_A_F)
            S_A_F = self.Resample(S_A_F)
            S_A_F = ECG.remove_dc_component(S_A_F)
            S_A_F = ECG.Normalize(S_A_F)
            S_A_F = self.Auto_correlation(S_A_F, S_A_F)
            O_C.append(S_A_F)
            S_A_F = self.compute_preserve_coff(S_A_F)
            A_P.append(S_A_F)
            S_A_F = DSP_TEST.DCTTransform.dct_transform(S_A_F)
            D_A.append(S_A_F)


        num_signals_O = len(O_C)
        fig, axs = plt.subplots(num_signals_O, 1, figsize=(8, 8 * num_signals_O))
        for idx, O in enumerate(O_C):
            O_index = [i for i in range(len(O))]
            axs[idx].plot(O_index, O, 'r-')
            axs[idx].scatter(O_index, O, color='green', marker='o')
            axs[idx].set_title(f' after auto correlation A  {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()

        num_signals_A_P = len(A_P)
        fig, axs = plt.subplots(num_signals_A_P, 1, figsize=(8, 8 * num_signals_A_P))
        for idx, P in enumerate(A_P):
            P_index = [i for i in range(len(P))]
            axs[idx].plot(P_index, P, 'r-')
            axs[idx].scatter(P_index, P, color='green', marker='o')
            axs[idx].set_title(f' after preserving A  {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()

        num_signals_D = len(D_A)
        fig, axs = plt.subplots(num_signals_D, 1, figsize=(8, 8 * num_signals_D))
        for idx, D in enumerate(D_A):
            D_index = [i for i in range(len(D))]
            axs[idx].plot(D_index, D, 'r-')
            axs[idx].scatter(D_index, D, color='green', marker='o')
            axs[idx].set_title(f' after DCT A  {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()

        print("FIRE A")

        O_C_B = []
        A_P_B= []
        D_A_B = []

        for S_B_F in B_signals:
            B_index = [i for i in range(len(S_B_F))]
            S_B_F = self.fillter(S_B_F)
            S_B_F = self.Resample(S_B_F)
            S_B_F = ECG.remove_dc_component(S_B_F)
            S_B_F = ECG.Normalize(S_B_F)
            S_B_F = self.Auto_correlation(S_B_F, S_B_F)
            O_C_B.append(S_B_F)
            S_B_F = self.compute_preserve_coff(S_B_F)
            A_P_B.append(S_B_F)
            S_B_F = DSP_TEST.DCTTransform.dct_transform(S_B_F)
            D_A_B.append(S_B_F)

        num_signals_O_C_B = len(O_C_B)
        fig, axs = plt.subplots(num_signals_O_C_B, 1, figsize=(8, 8 * num_signals_O_C_B))
        for idx, OO in enumerate(O_C_B):
            OO_index = [i for i in range(len(OO))]
            axs[idx].plot(OO_index, OO, 'r-')
            axs[idx].scatter(OO_index, OO, color='green', marker='o')
            axs[idx].set_title(f' after auto correlation B {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()

        num_signals_A_P_B = len(A_P_B)
        fig, axs = plt.subplots(num_signals_A_P_B, 1, figsize=(8, 8 * num_signals_A_P_B))
        for idx, PP in enumerate(A_P_B):
            PP_index = [i for i in range(len(PP))]
            axs[idx].plot(PP_index, PP, 'r-')
            axs[idx].scatter(PP_index, PP, color='green', marker='o')
            axs[idx].set_title(f' after preserving B  {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()

        num_signals_D_A_B = len(D_A_B)
        fig, axs = plt.subplots(num_signals_D_A_B, 1, figsize=(8, 8 * num_signals_D_A_B))
        for idx, DD in enumerate(D_A_B):
            D_A_B_index = [i for i in range(len(DD))]
            axs[idx].plot(D_A_B_index, DD, 'r-')
            axs[idx].scatter(D_A_B_index, DD, color='green', marker='o')
            axs[idx].set_title(f' after DCT B {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()
        print("FIRE B")

        O_C_T = []
        A_P_T = []
        D_A_T = []

        for S_T_F in Test_signals:
            Test_index = [i for i in range(len(S_T_F))]
            S_T_F = self.fillter(S_T_F)
            S_T_F = self.Resample(S_T_F)
            S_T_F = ECG.remove_dc_component(S_T_F)
            S_T_F = ECG.Normalize(S_T_F)
            S_T_F = self.Auto_correlation(S_T_F, S_T_F)
            O_C_T.append(S_T_F)
            S_T_F = self.compute_preserve_coff(S_T_F)
            A_P_T.append(S_T_F)
            S_T_F = DSP_TEST.DCTTransform.dct_transform(S_T_F)
            D_A_T.append(S_T_F)

        num_signals_O_C_T = len(O_C_T)
        fig, axs = plt.subplots(num_signals_O_C_T, 1, figsize=(8, 8 * num_signals_O_C_T))
        for idx, OT in enumerate(O_C_T):
            OT_index = [i for i in range(len(OT))]
            axs[idx].plot(OT_index, OT, 'r-')
            axs[idx].scatter(OT_index, OT, color='green', marker='o')
            axs[idx].set_title(f' after auto correlation Test {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()

        num_signals_A_P_T = len(A_P_T)
        fig, axs = plt.subplots(num_signals_A_P_T, 1, figsize=(8, 8 * num_signals_A_P_T))
        for idx, PT in enumerate(A_P_T):
            PT_index = [i for i in range(len(PT))]
            axs[idx].plot(PT_index, PT, 'r-')
            axs[idx].scatter(PT_index, PT, color='green', marker='o')
            axs[idx].set_title(f' after preserving Test  {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()

        num_signals_D_A_T = len(D_A_T)
        fig, axs = plt.subplots(num_signals_D_A_T, 1, figsize=(8, 8 * num_signals_D_A_T))
        for idx, DT in enumerate(D_A_T):
            DT_index = [i for i in range(len(DT))]
            axs[idx].plot(DT_index, DT, 'r-')
            axs[idx].scatter(DT_index, DT, color='green', marker='o')
            axs[idx].set_title(f' after DCT Test {idx + 1} - Continuous Plot')
            axs[idx].set_xlabel('X')
            axs[idx].set_ylabel('Y')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()
        print("FIRE Test")

        class1_templates = self.compute_templates(A_signals)
        class2_templates = self.compute_templates(B_signals)

        predictions = []
        for test_signal in Test_signals:
            predicted_class = self.template_matching(test_signal, class1_templates + class2_templates)
            predictions.append(predicted_class)

        for i, predicted_class in enumerate(predictions):
            if predicted_class == 1:
                test_message = f"Test Signal {i + 1} Class B "

                QtWidgets.QMessageBox.information(self, 'Test Result', test_message)
            else:
                test_message = f"Test Signal {i + 1} Class A "
                QtWidgets.QMessageBox.information(self, 'Test Result', test_message)

    def fillter(self, orginal_signal):
        value1 = self.miniF_textbox1.text()
        try:
            miniF = int(value1)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for miniF.')
            return

        value2 = self.maxF_textbox1.text()
        try:
            maxF = int(value2)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for maxF.')
            return

        value3 = self.FS_textbox1.text()
        try:
            FS = int(value3)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for FS.')
            return
        signal_index = [i for i in range(len(orginal_signal))]
        Stop_Band = 50
        Transition_Band = 500
        f1 = miniF
        f2 = maxF
        if miniF > 0 and maxF < FS / 2:
            filter_type = 'Band pass'
        else:
            filter_type = 'Band Stop'

        bands = [21, 44, 53, 74]
        window_names = ['rectangular', 'hanning', 'hamming', 'blackman']
        for i in range(len(bands)):
            if bands[i] >= Stop_Band:
                type_of_window = window_names[i]
                break

        normalized_transition_band = Transition_Band / FS
        if type_of_window == 'rectangular':
            total_elements = self.next_odd_num(0.9 / normalized_transition_band)

        elif type_of_window == 'hanning':
            total_elements = self.next_odd_num(3.1 / normalized_transition_band)

        elif type_of_window == 'hamming':
            total_elements = self.next_odd_num(3.3 / normalized_transition_band)

        else:
            total_elements = self.next_odd_num(5.5 / normalized_transition_band)

        if filter_type == 'Low pass':
            new_fc = (f1 + (Transition_Band / 2)) / FS, None

        elif filter_type == 'High pass':
            new_fc = (f1 - (Transition_Band / 2)) / FS, None

        elif filter_type == 'Band pass':
            new_fc = (f1 - (Transition_Band / 2)) / FS, (f2 + (Transition_Band / 2)) / FS

        else:
            new_fc = (f1 + (Transition_Band / 2)) / FS, (f2 - (Transition_Band / 2)) / FS

        list_1 = []
        list_2 = []
        for i in range((total_elements / 2).__ceil__()):
            list_1.append(i)
            list_2.append(-i)
        indicates = list_1 + list_2
        indicates = list(set(indicates))
        indicates.sort()

        windows_list = []
        for element_index in range((total_elements / 2).__ceil__()):
            if type_of_window == 'hanning':
                windows_list.append(0.5 + (0.5 * math.cos((2 * math.pi * element_index) / total_elements)))
            elif type_of_window == 'hamming':
                windows_list.append(0.54 + (0.46 * math.cos((2 * math.pi * element_index) / total_elements)))
            elif type_of_window == 'blackman':
                first_element = 0.5 * math.cos((2 * math.pi * element_index) / (total_elements - 1))
                second_element = 0.08 * math.cos((4 * math.pi * element_index) / (total_elements - 1))
                windows_list.append(0.42 + second_element + first_element)

        f1 = new_fc[0]
        f2 = new_fc[1]
        filtered_list = []
        for element_index in range((total_elements / 2).__ceil__()):
            if filter_type == 'Low pass':
                if element_index == 0:
                    filtered_list.append(2 * f1)
                else:
                    x = element_index * 2 * math.pi * f1
                    filtered_list.append(2 * f1 * (math.sin(x) / x))
            elif filter_type == 'High pass':
                if element_index == 0:
                    filtered_list.append(1 - (2 * f1))
                else:
                    x = element_index * 2 * math.pi * f1
                    filtered_list.append(-2 * f1 * (math.sin(x) / x))
            elif filter_type == 'Band pass':
                if element_index == 0:
                    filtered_list.append(2 * round(f2 - f1, 2))
                else:
                    x_1 = element_index * 2 * math.pi * f1
                    x_2 = element_index * 2 * math.pi * f2
                    filtered_list.append((2 * f2 * (math.sin(x_2) / x_2)) - (2 * f1 * (math.sin(x_1) / x_1)))
            else:
                if element_index == 0:
                    filtered_list.append(1 - (2 * (f2 - f1)))
                else:
                    x__1 = element_index * 2 * math.pi * f1
                    x__2 = element_index * 2 * math.pi * f2
                    filtered_list.append(
                        ((2 * f2 * (math.sin(x__2) / x__2)) - (2 * f1 * (math.sin(x__1) / x__1))) * -1)

        list1 = [x * y for x, y in zip(windows_list, filtered_list)]
        list2 = [x * y for x, y in zip(windows_list, filtered_list)]
        list2.reverse()
        list2.extend(list1)
        list2.remove(list2[int(len(list2) / 2)])

        len_signal_1 = len(list2)
        len_signal_2 = len(orginal_signal)
        len_output_signal = len_signal_1 + len_signal_2 - 1
        output_signal = [0] * len_output_signal
        for n in range(len_output_signal):
            for k in range(max(0, n - len_signal_2 + 1), min(len_signal_1, n + 1)):
                output_signal[n] += list2[k] * orginal_signal[n - k]
        output_indexes = indicates + signal_index
        x = list(set(output_indexes))
        x.sort()
        return output_signal

    def Resample(self, orginal_signal):
        value1 = self.miniF_textbox1.text()
        try:
            miniF = int(value1)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for miniF.')
            return

        value2 = self.maxF_textbox1.text()
        try:
            maxF = int(value2)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for maxF.')
            return

        value3 = self.FS_textbox1.text()
        try:
            FS = int(value3)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for FS.')
            return
        value4 = self.NewFS_textbox1.text()
        try:
            New_FS = int(value4)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid integer for New FS.')
            return
        signal_index = [i for i in range(len(orginal_signal))]
        Stop_Band = 50
        Transition_Band = 500
        f1 = miniF
        f2 = maxF
        if miniF > 0 and maxF < FS / 2:
            filter_type = 'Band pass'
        else:
            filter_type = 'Band Stop'
        if New_FS >= 2 * maxF:
            # up
            if New_FS > FS:

                upsampled_signal = []
                factor = int(New_FS / FS)

                for i in orginal_signal:
                    upsampled_signal.extend([i] + [0] * (factor - 1))

                bands = [21, 44, 53, 74]
                window_names = ['rectangular', 'hanning', 'hamming', 'blackman']
                for i in range(len(bands)):
                    if bands[i] >= Stop_Band:
                        type_of_window = window_names[i]
                        break

                normalized_transition_band = Transition_Band / FS
                if type_of_window == 'rectangular':
                    total_elements = self.next_odd_num(0.9 / normalized_transition_band)

                elif type_of_window == 'hanning':
                    total_elements = self.next_odd_num(3.1 / normalized_transition_band)

                elif type_of_window == 'hamming':
                    total_elements = self.next_odd_num(3.3 / normalized_transition_band)

                else:
                    total_elements = self.next_odd_num(5.5 / normalized_transition_band)

                if filter_type == 'Low pass':
                    new_fc = (f1 + (Transition_Band / 2)) / FS, None

                elif filter_type == 'High pass':
                    new_fc = (f1 - (Transition_Band / 2)) / FS, None

                elif filter_type == 'Band pass':
                    new_fc = (f1 - (Transition_Band / 2)) / FS, (f2 + (Transition_Band / 2)) / FS

                else:
                    new_fc = (f1 + (Transition_Band / 2)) / FS, (f2 - (Transition_Band / 2)) / FS

                list_1 = []
                list_2 = []
                for i in range((total_elements / 2).__ceil__()):
                    list_1.append(i)
                    list_2.append(-i)
                indicates = list_1 + list_2
                indicates = list(set(indicates))
                indicates.sort()

                windows_list = []
                for element_index in range((total_elements / 2).__ceil__()):
                    if type_of_window == 'hanning':
                        windows_list.append(0.5 + (0.5 * math.cos((2 * math.pi * element_index) / total_elements)))
                    elif type_of_window == 'hamming':
                        windows_list.append(
                            0.54 + (0.46 * math.cos((2 * math.pi * element_index) / total_elements)))
                    elif type_of_window == 'blackman':
                        first_element = 0.5 * math.cos((2 * math.pi * element_index) / (total_elements - 1))
                        second_element = 0.08 * math.cos((4 * math.pi * element_index) / (total_elements - 1))
                        windows_list.append(0.42 + second_element + first_element)

                f1 = new_fc[0]
                f2 = new_fc[1]
                filtered_list = []
                for element_index in range((total_elements / 2).__ceil__()):
                    if filter_type == 'Low pass':
                        if element_index == 0:
                            filtered_list.append(2 * f1)
                        else:
                            x = element_index * 2 * math.pi * f1
                            filtered_list.append(2 * f1 * (math.sin(x) / x))
                    elif filter_type == 'High pass':
                        if element_index == 0:
                            filtered_list.append(1 - (2 * f1))
                        else:
                            x = element_index * 2 * math.pi * f1
                            filtered_list.append(-2 * f1 * (math.sin(x) / x))
                    elif filter_type == 'Band pass':
                        if element_index == 0:
                            filtered_list.append(2 * round(f2 - f1, 2))
                        else:
                            x_1 = element_index * 2 * math.pi * f1
                            x_2 = element_index * 2 * math.pi * f2
                            filtered_list.append(
                                (2 * f2 * (math.sin(x_2) / x_2)) - (2 * f1 * (math.sin(x_1) / x_1)))
                    else:
                        if element_index == 0:
                            filtered_list.append(1 - (2 * (f2 - f1)))
                        else:
                            x__1 = element_index * 2 * math.pi * f1
                            x__2 = element_index * 2 * math.pi * f2
                            filtered_list.append(
                                ((2 * f2 * (math.sin(x__2) / x__2)) - (2 * f1 * (math.sin(x__1) / x__1))) * -1)

                list1 = [x * y for x, y in zip(windows_list, filtered_list)]
                list2 = [x * y for x, y in zip(windows_list, filtered_list)]
                list2.reverse()
                list2.extend(list1)
                list2.remove(list2[int(len(list2) / 2)])

                len_signal_1 = len(list2)
                len_signal_2 = len(upsampled_signal)
                len_output_signal = len_signal_1 + len_signal_2 - 1
                output_signal = [0] * len_output_signal
                for n in range(len_output_signal):
                    for k in range(max(0, n - len_signal_2 + 1), min(len_signal_1, n + 1)):
                        output_signal[n] += list2[k] * upsampled_signal[n - k]
                output_indexes = indicates + signal_index
                x = list(set(output_indexes))
                x.sort()
                for i in range(2):
                    output_signal.remove(output_signal[-1])
                    x.remove(x[-1])
                    return output_signal


            # down
            elif New_FS < FS:
                bands = [21, 44, 53, 74]
                window_names = ['rectangular', 'hanning', 'hamming', 'blackman']
                for i in range(len(bands)):
                    if bands[i] >= Stop_Band:
                        type_of_window = window_names[i]
                        break

                normalized_transition_band = Transition_Band / FS
                if type_of_window == 'rectangular':
                    total_elements = self.next_odd_num(0.9 / normalized_transition_band)

                elif type_of_window == 'hanning':
                    total_elements = self.next_odd_num(3.1 / normalized_transition_band)

                elif type_of_window == 'hamming':
                    total_elements = self.next_odd_num(3.3 / normalized_transition_band)

                else:
                    total_elements = self.next_odd_num(5.5 / normalized_transition_band)

                if filter_type == 'Low pass':
                    new_fc = (f1 + (Transition_Band / 2)) / FS, None

                elif filter_type == 'High pass':
                    new_fc = (f1 - (Transition_Band / 2)) / FS, None

                elif filter_type == 'Band pass':
                    new_fc = (f1 - (Transition_Band / 2)) / FS, (f2 + (Transition_Band / 2)) / FS

                else:
                    new_fc = (f1 + (Transition_Band / 2)) / FS, (f2 - (Transition_Band / 2)) / FS

                list_1 = []
                list_2 = []
                for i in range((total_elements / 2).__ceil__()):
                    list_1.append(i)
                    list_2.append(-i)
                indicates = list_1 + list_2
                indicates = list(set(indicates))
                indicates.sort()

                windows_list = []
                for element_index in range((total_elements / 2).__ceil__()):
                    if type_of_window == 'hanning':
                        windows_list.append(0.5 + (0.5 * math.cos((2 * math.pi * element_index) / total_elements)))
                    elif type_of_window == 'hamming':
                        windows_list.append(
                            0.54 + (0.46 * math.cos((2 * math.pi * element_index) / total_elements)))
                    elif type_of_window == 'blackman':
                        first_element = 0.5 * math.cos((2 * math.pi * element_index) / (total_elements - 1))
                        second_element = 0.08 * math.cos((4 * math.pi * element_index) / (total_elements - 1))
                        windows_list.append(0.42 + second_element + first_element)

                f1 = new_fc[0]
                f2 = new_fc[1]
                filtered_list = []
                for element_index in range((total_elements / 2).__ceil__()):
                    if filter_type == 'Low pass':
                        if element_index == 0:
                            filtered_list.append(2 * f1)
                        else:
                            x = element_index * 2 * math.pi * f1
                            filtered_list.append(2 * f1 * (math.sin(x) / x))
                    elif filter_type == 'High pass':
                        if element_index == 0:
                            filtered_list.append(1 - (2 * f1))
                        else:
                            x = element_index * 2 * math.pi * f1
                            filtered_list.append(-2 * f1 * (math.sin(x) / x))
                    elif filter_type == 'Band pass':
                        if element_index == 0:
                            filtered_list.append(2 * round(f2 - f1, 2))
                        else:
                            x_1 = element_index * 2 * math.pi * f1
                            x_2 = element_index * 2 * math.pi * f2
                            filtered_list.append(
                                (2 * f2 * (math.sin(x_2) / x_2)) - (2 * f1 * (math.sin(x_1) / x_1)))
                    else:
                        if element_index == 0:
                            filtered_list.append(1 - (2 * (f2 - f1)))
                        else:
                            x__1 = element_index * 2 * math.pi * f1
                            x__2 = element_index * 2 * math.pi * f2
                            filtered_list.append(
                                ((2 * f2 * (math.sin(x__2) / x__2)) - (2 * f1 * (math.sin(x__1) / x__1))) * -1)

                list1 = [x * y for x, y in zip(windows_list, filtered_list)]
                list2 = [x * y for x, y in zip(windows_list, filtered_list)]
                list2.reverse()
                list2.extend(list1)
                list2.remove(list2[int(len(list2) / 2)])

                len_signal_1 = len(list2)
                len_signal_2 = len(orginal_signal)
                len_output_signal = len_signal_1 + len_signal_2 - 1
                output_signal = [0] * len_output_signal
                for n in range(len_output_signal):
                    for k in range(max(0, n - len_signal_2 + 1), min(len_signal_1, n + 1)):
                        output_signal[n] += list2[k] * orginal_signal[n - k]
                output_indexes = indicates + signal_index
                x = list(set(output_indexes))
                x.sort()

                down_sampled_signal = []
                down_sampled_signal_indicates = []
                M = int((FS / New_FS))
                for i in range(0, len(output_signal), M):
                    down_sampled_signal.append(output_signal[i])
                for i in range(len(down_sampled_signal)):
                    down_sampled_signal_indicates.append(x[i])
                return down_sampled_signal


            else:
                QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'ERROR')
        else:
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'ERROR')

    def Auto_correlation(self, signal1, signal2):
        correlation = []
        for i in range(len(signal2)):
            value = self.calculate_cross_correlation_element(signal1, signal2) / len(signal2)
            signal2 = self.shift_signal(signal2)
            correlation.append(value)
        return correlation

    @staticmethod
    def shift_signal(signal):
        tmp_value = signal[0]
        signal = signal[1:]
        signal.append(tmp_value)
        return signal

    @staticmethod
    def calculate_mean_of_signal(signal_values):
        summ = 0
        len_of_values = len(signal_values)
        for i in range(len_of_values):
            summ += signal_values[i]
        return summ / len_of_values

    @staticmethod
    def Normalize(orginal_signal):
        min_value = min(orginal_signal)
        max_value = max(orginal_signal)
        signal_index = [i for i in range(len(orginal_signal))]
        signal_output = [2 * ((x - min_value) / (max_value - min_value)) - 1 for x in orginal_signal]
        # i = (orginal_signal - ECG.calculate_mean_of_signal(orginal_signal)) / max(abs(orginal_signal))
        # fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        # axs[0].stem(signal_index, i, linefmt='b-', markerfmt='bo', basefmt='r-')
        # axs[0].set_title('Normalize Result - Discrete Plot')
        # axs[0].set_xlabel('X')
        # axs[0].set_ylabel('Y')
        # axs[0].grid(True)
        # axs[1].plot(signal_index, i, 'r-')
        # axs[1].scatter(signal_index, i, color='red', marker='o')
        # axs[1].set_title('Normalize Result - Continuous Plot')
        # axs[1].set_xlabel('X')
        # axs[1].set_ylabel('Y')
        # axs[1].grid(True)
        # plt.tight_layout()
        # plt.show()
        return signal_output

    @staticmethod
    def find_peaks(signal):
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                peaks.append(i)
        return peaks

    @staticmethod
    def compute_preserve_coff(signal):
        signal = signal[len(signal) // 2:]
        peaks = ECG.find_peaks(signal)
        return peaks

    @staticmethod
    def calculate_cross_correlation_element(signal1, signal2):
        summ = 0
        for i in range(len(signal2)):
            summ += signal1[i] * signal2[i]
        return summ

    @staticmethod
    def remove_dc_component(signal_values):
        len_of_values = len(signal_values)
        removed_values = []
        for i in range(len_of_values):
            result = signal_values[i] - ECG.calculate_mean_of_signal(signal_values)
            removed_values.append(round(result, 3))
        return removed_values

    @staticmethod
    def load_signals_from_folder(folder_path):
        signals = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                _, _, _, S = ECG.read_signal_from_file_without_indicates(file_path)
                signals.append(S)
        return signals

    @staticmethod
    def read_signal_from_file_without_indicates(file_name):
        signal = open(file_name)
        # define the signal
        signal_type = float(signal.readline().strip())
        is_periodic = float(signal.readline().strip())
        num_samples = float(signal.readline().strip())
        samples = [list(map(float, line.strip().split())) for line in signal]
        values = [sample[0] for sample in samples]
        return signal_type, is_periodic, num_samples, values

    @staticmethod
    def compute_templates(class_signals):
        return [ECG.calculate_mean_of_signal(Y) for Y in class_signals]

    @staticmethod
    def C_DCT(final_signal):
        N = len(final_signal)
        dct_result = np.zeros_like(final_signal, dtype=float)
        for k in range(N):
            x_values[k] = 0
            sum_val = 0.0
            for n in range(N):
                sum_val += final_signal[n] * np.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
            dct_result[k] = np.sqrt(2 / N) * sum_val

        return dct_result

    def template_matching(self, test_signal, templates):
        correlations = self.Auto_correlation(test_signal, templates)
        predicted_class = np.argmax(correlations)
        return predicted_class
