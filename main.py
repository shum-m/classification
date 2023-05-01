import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

NAME_CSV = 'data.csv'
"""Путь к .csv файлу"""

LEARNING_PERCENT = 0.1
"""Размер обучающей выборки."""

ITERATIONS = 100
"""Число итераций."""

LEARNING_RATE = 10 ** -5
"""Коэффициент обучения."""

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
    def calc_cost(m, out_params, weights):
        """
        Считает функцию потерь в методе градиентного спуска.
        :param m: Размерность.
        :param out_params: Выходные параметры.
        :param weights: Веса.
        :return: Функция потерь.
        """
        return -(1 / m) * np.sum(out_params * (np.log(weights)) + (1 - out_params) * (np.log(1 - weights)))

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
    def update_weights(m, weights, in_params, out_params):
        """
        Обновление весов
        :param m: Размерность.
        :param weights: Веса.
        :param in_params: Входные данные.
        :param out_params: Выходные данные.
        :return: Новые веса.
        """
        dw = (1 / m) * np.dot(in_params, (weights - out_params).T)
        return dw

    @staticmethod
    def gradient_descent(w, in_params, out_params, num_iterations, learning_rate):
        """
        Метод градиентного спуска.
        :param w: Веса.
        :param in_params: Входные данные.
        :param out_params: Выходные данные.
        :param num_iterations: Количество итераци.
        :param learning_rate: Коэффициент обучения.
        :return: Веса, новые веса, функция ошибок.
        """

        costs = []
        m = in_params.shape[1]
        dw = None

        for i in range(num_iterations):
            new_weights = MyLogisticRegression.calc_weights(w, in_params)
            cost = MyLogisticRegression.calc_cost(m, out_params, new_weights)

            dw = MyLogisticRegression.update_weights(m, new_weights, in_params, out_params)
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
        :return: Прогноз, точность прогноза.
        """
        m = in_params.shape[1]
        weights = np.dot(w.T, in_params)
        new_prediction = np.zeros((1, m))

        accuracy_count = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for i in range(1, m):
            if weights[0][i] <= 0.5:
                new_prediction[0][i] = 0
            else:
                new_prediction[0][i] = 1

        for i in range(1, m):
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

        return new_prediction, (accuracy_count / in_params.shape[1]) * 100

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
        :return: Прогноз для тренировочной выборки, точность этого прогноза,
        прогноз для тестовой выборки, точность для этого прогноза.
        """
        w = np.zeros([in_train.shape[0], 1])

        w, dw, costs = MyLogisticRegression.gradient_descent(w, in_train, out_train, num_iterations, learning_rate)

        train_predict, train_accuracy = MyLogisticRegression.prediction(w, in_train, out_train)
        test_predict, test_accuracy = MyLogisticRegression.prediction(w, in_test, out_test)

        return train_predict, train_accuracy, test_predict, test_accuracy, costs


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
        print(self.data.shape)

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

        print(self.data.head)

    def get_train_test_samples(self):
        """
        Получает выборки для обучения и тестирования.
        :return: Датафреймы для обучения и тестирования.
        """
        train_data = self.data.sample(frac=LEARNING_PERCENT).sort_values(by=[self.cols[COL_INDEX_NUM_CLIENT]])
        print('Обучение:')
        print(train_data.head)

        test_data = self.data.drop(train_data.index).sort_values(by=[self.cols[COL_INDEX_NUM_CLIENT]])
        print('Тестирование:')
        print(test_data.head)

        return train_data, test_data

    @staticmethod
    def get_input_output_data(dataframe):
        """
        Получает выходные и выходные данные.
        :param dataframe: Выборка данных.
        :return: Датафреймы входных и выходных значений.
        """
        input_data = dataframe[dataframe.columns[COL_INDEX_SCORE]].to_numpy().T
        output_data = dataframe[dataframe.columns[COL_INDEX_DEBT]].to_numpy().T

        return input_data, output_data

    @staticmethod
    def generate_new_data(train_dataframe, result_train, test_dataframe, result_test):
        """
        Генерирует новые данные: <file_name>_result.csv - данные с прогнозом;
        <file_name>_errors.csv - данные об ошибках I и II рода
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

        new_dataframe.to_csv(NAME_CSV.replace('.csv', '_result.csv'), sep=';', encoding='windows-1251')

        # TODO: ДОБАВИТЬ ВЫЧИСЛЕНИЕ ОШИБОК ПЕРВОГО И ВТОРОГО РОДА


if __name__ == '__main__':
    d = BankData()

    d.convert_to_analyse()
    data_train, data_test = d.get_train_test_samples()

    # Логистическая регрессия

    # надо ли использовать столбец баллов БКИ при лог. регрессии?
    input_train, output_train = BankData.get_input_output_data(data_train)
    input_test, output_test = BankData.get_input_output_data(data_test)

    print('Обучающие входные и выходные')
    print(input_train, output_train)

    print('Тестовые входные и выходные')
    print(input_test, output_test)

    # Логистическая регрессия MyLogisticRegression

    train_pr_mlr, train_a_mlr, test_pr_mlr, test_a_mlr, costs_f = \
        MyLogisticRegression.logistic_regression(
            input_train.reshape(-1, 1).T,
            output_train.reshape(-1, 1).T,
            input_test.reshape(-1, 1).T,
            output_test.reshape(-1, 1).T)

    print('(MyLogisticRegression) Точность на обучающей выборке ' + str(train_a_mlr) + ' %')
    print('(MyLogisticRegression) Точность на тестовой выборке ' + str(test_a_mlr) + ' %')

    # Логистическая регрессия sklearn
    clf = LogisticRegression(random_state=0, max_iter=ITERATIONS).fit(input_train.reshape(-1, 1), output_train.T)
    train_a_skl = clf.score(input_train.reshape(-1, 1), output_train.T)
    print('(SKLEARN) Точность на обучающей выборке ' + str(train_a_skl * 100) + ' %')

    y_pred = clf.predict(input_test.reshape(-1, 1))
    test_a_skl = clf.score(input_test.reshape(-1, 1), output_test)
    print('(SKLEARN) Точность на тестовой выборке ' + str(test_a_skl * 100) + ' %')

    # Создание новых файлов
    d.generate_new_data(data_train, train_pr_mlr, data_test, test_pr_mlr)
