import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

NAME_CSV = 'data.csv'
"""Путь к .csv файлу."""

LEARNING_PERCENT = 0.8
"""Размер обучающей выборки."""

ITERATIONS = 150
"""Число итераций."""

LEARNING_RATE = 5 * 10 ** -4
"""Коэффициент обучения."""

LOGISTIC_REGRESSION_THRESHOLD = 0.5
"""Пороговое значение логистической регрессии."""

COL_INDEX_NUM_CLIENT = 0
"""Индекс столбца номера клиента банка."""

COL_INDEX_ID_CLIENT = 1
"""Индекс столбца id клиента банка."""

COL_INDEX_BKI_SCORE = 2
"""Индекс столбца скорингового балла БКИ."""

COL_INDEX_SCORE = 3
"""Индекс столбца скорингового банковского скорингового балла."""

COL_INDEX_DEBT = 4
"""Индекс столбца задолженности 60 дней."""

NAME_PREDICT_COLUMN = 'Спрогнозированные значения'
"""Имя столбца со спрогнозированными данными."""


class MyLogisticRegression:
    """Логистическая регрессия."""

    @staticmethod
    def sigmoid(a):
        """
        Сигмоидная функция.
        :param a: Значение аргумента.
        :return: Значение функции.
        """
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def calc_cost(m, out_params, comb):
        """
        Считает функцию потерь в методе градиентного спуска.
        :param m: Размерность.
        :param out_params: Выходные параметры.
        :param comb: Комбинация.
        :return: Функция потерь.
        """
        return -(1 / m) * np.sum(out_params * (np.log(comb)) + (1 - out_params) * (np.log(1 - comb)))

    @staticmethod
    def calc_weights(w, in_params):
        """
        Вычисление весов.
        :param w: Старые веса.
        :param in_params: Входные данные.
        :return: Новые веса.
        """
        weights = MyLogisticRegression.sigmoid(np.dot(w.T, in_params))
        return weights

    @staticmethod
    def update_weights(m, comb, in_params, out_params):
        """
        Обновление весов
        :param m: Размерность.
        :param comb: Комбинация..
        :param in_params: Входные данные.
        :param out_params: Выходные данные.
        :return: Новые веса.
        """
        dw = (1 / m) * np.dot(in_params, (comb - out_params).T)
        return dw

    @staticmethod
    def gradient_descent(w, in_params, out_params, num_iterations, learning_rate):
        """
        Метод градиентного спуска.
        :param w: Веса.
        :param in_params: Входные данные.
        :param out_params: Выходные данные.
        :param num_iterations: Количество итераций.
        :param learning_rate: Коэффициент обучения.
        :return: Веса, новые веса, функция ошибок.
        """

        costs = []
        m = in_params.shape[1]
        dw = None

        for i in range(num_iterations):
            comb = MyLogisticRegression.calc_weights(w, in_params)
            cost = MyLogisticRegression.calc_cost(m, out_params, comb)

            dw = MyLogisticRegression.update_weights(m, comb, in_params, out_params)
            w = w - learning_rate * dw

            costs.append(cost)

        return w, dw, costs

    @staticmethod
    def prediction(w, in_params, out_params):
        """
        Делает прогноз и вычисляет точность.
        :param w: Веса.
        :param in_params: Входные параметры.
        :param out_params: Выходные параметры.
        :return: Прогноз, точность прогноза, матрица сопряженности, вероятности.
        """
        m = in_params.shape[1]
        comb = np.dot(w.T, in_params)
        new_prediction = np.zeros((1, m))
        proba = []
        accuracy_count = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for i in range(m):
            proba.append(MyLogisticRegression.sigmoid(comb[0][i]))
            if proba[i] <= LOGISTIC_REGRESSION_THRESHOLD:
                new_prediction[0][i] = 0
            else:
                new_prediction[0][i] = 1

            if new_prediction[0][i] == out_params[0][i]:
                accuracy_count += 1

            if new_prediction[0][i] == out_params[0][i] == 1:
                true_positive += 1
            elif new_prediction[0][i] == out_params[0][i] == 0:
                true_negative += 1

            if new_prediction[0][i] != out_params[0][i] and new_prediction[0][i] == 0 and out_params[0][i] == 1:
                false_negative += 1
            elif new_prediction[0][i] != out_params[0][i] and new_prediction[0][i] == 1 and out_params[0][i] == 0:
                false_positive += 1

        confusion_matrix = np.array([true_positive, true_negative, false_positive, false_negative]).reshape((2, 2))
        result_prediction = {'prediction': new_prediction, 'accuracy': (accuracy_count / in_params.shape[1]) * 100,
                             'confusion': confusion_matrix, 'proba': np.array(proba)}

        return result_prediction

    @staticmethod
    def logistic_regression(in_train, out_train, in_test, out_test, learning_rate=LEARNING_RATE,
                            num_iterations=ITERATIONS):
        """
        Вычисления логистической регрессии для обучающей и тестовой выборок.
        :param in_train: Тренировочные входные данные.
        :param out_train: Тренировочные выходные данные.
        :param in_test: Тестовые входные данные.
        :param out_test: Тестовые выходные данные.
        :param learning_rate: Коэффициент обучения.
        :param num_iterations: Количество итераций.
        :return: Для каждой выборки: прогноз для выборки, точность этого прогноза, матрицу сопряженности
        (ошибки 1 и 2 рода), вероятности, функция ошибок модели.
        """
        w = np.zeros([in_train.shape[0], 1])

        w, dw, costs = MyLogisticRegression.gradient_descent(w, in_train, out_train, num_iterations, learning_rate)

        train_logistic_regression = MyLogisticRegression.prediction(w, in_train, out_train)
        test_logistic_regression = MyLogisticRegression.prediction(w, in_test, out_test)

        return train_logistic_regression, test_logistic_regression


class BankData:
    """
    Банковские данные.
    """

    def __init__(self):
        """
        Получает данные из csv файла.
        """

        self.data = pd.read_csv(NAME_CSV, header=0, encoding='windows-1251', sep=';', decimal=',')
        """Данные из csv файла."""

        self.data.dropna()

        self.cols = list(self.data.columns)
        """Столбцы документа."""

    def convert_to_analyse(self):
        """
        Преобразует данные для анализа.
        :return: Данные без пустых строк, без строк с отрицательным скоринговым баллом, с целыми значения скорингового
        балла.
        """
        self.data = self.data.loc[self.data[self.cols[COL_INDEX_BKI_SCORE]] >= 0]
        self.data[self.cols[COL_INDEX_SCORE]] = self.data[self.cols[COL_INDEX_SCORE]].astype(int)

    def get_train_test_samples(self):
        """
        Получает выборки для обучения и тестирования.
        :return: Датафреймы для обучения и тестирования.
        """
        train_data = self.data.sample(frac=LEARNING_PERCENT, random_state=0).sort_values(
            by=[self.cols[COL_INDEX_NUM_CLIENT]])

        test_data = self.data.drop(train_data.index).sort_values(by=[self.cols[COL_INDEX_NUM_CLIENT]])

        return train_data, test_data

    @staticmethod
    def get_input_output_data(dataframe):
        """
        Получает выходные и выходные данные.
        :param dataframe: Выборка данных.
        :return: Датафреймы входных и выходных значений.
        """
        input_data = dataframe[dataframe.columns[COL_INDEX_SCORE]].to_numpy().T
        input_data = BankData.minimax_normalize(input_data)

        output_data = dataframe[dataframe.columns[COL_INDEX_DEBT]].to_numpy().T

        return input_data, output_data

    @staticmethod
    def minimax_normalize(not_norm_data):
        """
        Минимакс нормализация.
        :param not_norm_data: Ненормализованные данные.
        :return: Нормализованные данные.
        """
        min_vals = np.min(not_norm_data, axis=0)
        max_vals = np.max(not_norm_data, axis=0)

        return (not_norm_data - min_vals) / (max_vals - min_vals)

    @staticmethod
    def generate_new_data(file_name, train_dataframe, result_train, test_dataframe, result_test):
        """
        Генерирует csv файл.
        :param file_name: Имя файла.
        :param train_dataframe: Датафрейм тренировочных данных.
        :param result_train: Результаты прогнозирования на тренировочных данных.
        :param test_dataframe: Датафрейм тестовых данных.
        :param result_test: Результаты прогнозирования тестовых данных.
        """

        train_dataframe[NAME_PREDICT_COLUMN] = result_train[0].astype(int)
        test_dataframe[NAME_PREDICT_COLUMN] = result_test[0].astype(int)

        new_dataframe = pd.concat([train_dataframe, test_dataframe])

        new_dataframe = new_dataframe.sort_values(by=[new_dataframe.columns[COL_INDEX_NUM_CLIENT]])
        new_dataframe = new_dataframe.drop(columns=new_dataframe.columns[COL_INDEX_NUM_CLIENT])

        new_dataframe.to_csv(file_name, sep=';', encoding='windows-1251')


class ROCCurve:
    @staticmethod
    def show_curve(classes, classes_proba):
        fpr, tpr, thresholds = roc_curve(classes, classes_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == '__main__':
    d = BankData()

    d.convert_to_analyse()
    data_train, data_test = d.get_train_test_samples()

    input_train, output_train = BankData.get_input_output_data(data_train)
    input_test, output_test = BankData.get_input_output_data(data_test)

    print('Обучающие входные и выходные')
    print(input_train, output_train)

    print('Тестовые входные и выходные')
    print(input_test, output_test)

    # region Логистическая регрессия MyLogisticRegression

    train_mlr, test_mlr = MyLogisticRegression.logistic_regression(
        input_train.reshape(-1, 1).T,
        output_train.reshape(-1, 1).T,
        input_test.reshape(-1, 1).T,
        output_test.reshape(-1, 1).T)

    print('(MyLogisticRegression) Точность на обучающей выборке ' + str(train_mlr['accuracy']) + ' %')
    print('(MyLogisticRegression) Точность на тестовой выборке ' + str(test_mlr['accuracy']) + ' %')
    print('(MyLogisticRegression) Тестовая выборка матрица сопряженности', test_mlr['confusion'])

    ROCCurve.show_curve(output_test, test_mlr['proba'])

    # endregion

    # region Логистическая регрессия sklearn
    clf = LogisticRegression(random_state=0, max_iter=ITERATIONS).fit(input_train.reshape(-1, 1), output_train.T)

    train_pred = clf.predict(input_train.reshape(-1, 1))
    test_pred = clf.predict(input_test.reshape(-1, 1))
    train_a_skl = clf.score(input_train.reshape(-1, 1), output_train.T)

    print('(SKLEARN) Точность на обучающей выборке ' + str(train_a_skl * 100) + ' %')

    test_a_skl = clf.score(input_test.reshape(-1, 1), output_test)
    print('(SKLEARN) Точность на тестовой выборке ' + str(test_a_skl * 100) + ' %')

    y_score = clf.predict_proba(input_test.reshape(-1, 1))[:, 1]
    ROCCurve.show_curve(output_test, y_score)

    # endregion

    # Создание нового файла
    d.generate_new_data('mlr_data.csv', data_train, train_mlr['prediction'], data_test, test_mlr['prediction'])
    d.generate_new_data('scikit_data.csv', data_train, train_pred, data_test, test_pred)
