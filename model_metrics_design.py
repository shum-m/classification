from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(408, 590)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.precision_label = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.precision_label.setFont(font)
        self.precision_label.setObjectName("precision_label")
        self.verticalLayout.addWidget(self.precision_label)
        self.precision_data = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.precision_data.setFont(font)
        self.precision_data.setObjectName("precision_data")
        self.verticalLayout.addWidget(self.precision_data)
        self.accuracy_text = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.accuracy_text.setFont(font)
        self.accuracy_text.setObjectName("accuracy_text")
        self.verticalLayout.addWidget(self.accuracy_text)
        self.accuracy_data = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.accuracy_data.setFont(font)
        self.accuracy_data.setObjectName("accuracy_data")
        self.verticalLayout.addWidget(self.accuracy_data)
        self.recall_text = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.recall_text.setFont(font)
        self.recall_text.setObjectName("recall_text")
        self.verticalLayout.addWidget(self.recall_text)
        self.recall_data = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.recall_data.setFont(font)
        self.recall_data.setObjectName("recall_data")
        self.verticalLayout.addWidget(self.recall_data)
        self.f_measure_text = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.f_measure_text.setFont(font)
        self.f_measure_text.setObjectName("f_measure_text")
        self.verticalLayout.addWidget(self.f_measure_text)
        self.f_measure_data = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.f_measure_data.setFont(font)
        self.f_measure_data.setObjectName("f_measure_data")
        self.verticalLayout.addWidget(self.f_measure_data)
        self.auc_roc_text = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.auc_roc_text.setFont(font)
        self.auc_roc_text.setObjectName("auc_roc_text")
        self.verticalLayout.addWidget(self.auc_roc_text)
        self.auc_roc_data = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.auc_roc_data.setFont(font)
        self.auc_roc_data.setObjectName("auc_roc_data")
        self.verticalLayout.addWidget(self.auc_roc_data)
        self.confusion_matrix_text = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.confusion_matrix_text.setFont(font)
        self.confusion_matrix_text.setObjectName("confusion_matrix_text")
        self.verticalLayout.addWidget(self.confusion_matrix_text)
        self.true_positive_text = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.true_positive_text.setFont(font)
        self.true_positive_text.setObjectName("true_positive_text")
        self.verticalLayout.addWidget(self.true_positive_text)
        self.true_positive_data = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.true_positive_data.setFont(font)
        self.true_positive_data.setObjectName("true_positive_data")
        self.verticalLayout.addWidget(self.true_positive_data)
        self.true_negative_text = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.true_negative_text.setFont(font)
        self.true_negative_text.setObjectName("true_negative_text")
        self.verticalLayout.addWidget(self.true_negative_text)
        self.true_negative_data = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.true_negative_data.setFont(font)
        self.true_negative_data.setObjectName("true_negative_data")
        self.verticalLayout.addWidget(self.true_negative_data)
        self.false_negative_text = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.false_negative_text.setFont(font)
        self.false_negative_text.setObjectName("false_negative_text")
        self.verticalLayout.addWidget(self.false_negative_text)
        self.false_negative_data = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.false_negative_data.setFont(font)
        self.false_negative_data.setObjectName("false_negative_data")
        self.verticalLayout.addWidget(self.false_negative_data)
        self.false_positive_text = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.false_positive_text.setFont(font)
        self.false_positive_text.setObjectName("false_positive_text")
        self.verticalLayout.addWidget(self.false_positive_text)
        self.false_positive_data = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.false_positive_data.setFont(font)
        self.false_positive_data.setObjectName("false_positive_data")
        self.verticalLayout.addWidget(self.false_positive_data)
        self.show_roc_curve = QtWidgets.QPushButton(Dialog)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.show_roc_curve.setFont(font)
        self.show_roc_curve.setObjectName("show_roc_curve")
        self.verticalLayout.addWidget(self.show_roc_curve)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Метрики модели"))
        self.precision_label.setText(_translate("Dialog", "Точность (precision)"))
        self.precision_data.setText(_translate("Dialog", "0"))
        self.accuracy_text.setText(_translate("Dialog", "Точность (accuracy)"))
        self.accuracy_data.setText(_translate("Dialog", "0"))
        self.recall_text.setText(_translate("Dialog", "Полнота (recall)"))
        self.recall_data.setText(_translate("Dialog", "0"))
        self.f_measure_text.setText(_translate("Dialog", "F-мера"))
        self.f_measure_data.setText(_translate("Dialog", "0"))
        self.auc_roc_text.setText(_translate("Dialog", "AUC-ROC"))
        self.auc_roc_data.setText(_translate("Dialog", "0"))
        self.confusion_matrix_text.setText(_translate("Dialog", "Матрица ошибок:"))
        self.true_positive_text.setText(_translate("Dialog", "Истинно положительные случаи"))
        self.true_positive_data.setText(_translate("Dialog", "0"))
        self.true_negative_text.setText(_translate("Dialog", "Истинно отрицательные случаи"))
        self.true_negative_data.setText(_translate("Dialog", "0"))
        self.false_negative_text.setText(_translate("Dialog", "Ложно отрицательные случаи (ошибка 1 рода)"))
        self.false_negative_data.setText(_translate("Dialog", "0"))
        self.false_positive_text.setText(_translate("Dialog", "Ложно положительные случаи (ошибка 2 рода)"))
        self.false_positive_data.setText(_translate("Dialog", "0"))
        self.show_roc_curve.setText(_translate("Dialog", "Показать ROC кривую"))
