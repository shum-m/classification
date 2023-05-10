import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


NAME_CSV = 'data.csv'
"""Путь к .csv файлу."""

LEARNING_PERCENT = 0.8
"""Размер обучающей выборки."""

ITERATIONS = 150
"""Число итераций."""

LEARNING_RATE = 1 * 10 ** -2
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
"""Индекс столбца банковского скорингового балла."""

COL_INDEX_DEBT = 4
"""Индекс столбца задолженности 60 дней."""

NAME_PREDICT_COLUMN = 'Спрогнозированные значения'
"""Имя столбца со спрогнозированными данными."""

NAME_MODEL_PROBABILITIES_COLUMN = 'Вероятности модели'
"""Имя столбца для вероятностей модели"""


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
        :param comb: Линейная комбинация.
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
        Делает прогноз, вычисляет точность.
        :param w: Веса.
        :param in_params: Входные параметры.
        :param out_params: Выходные параметры.
        :return: Прогноз, точность прогноза (соотношение количества правильных прогнозов), вероятности.
        """
        m = in_params.shape[1]
        comb = np.dot(w.T, in_params)
        new_prediction = np.zeros((1, m))
        proba = []
        accuracy_count = 0

        for i in range(m):
            proba.append(MyLogisticRegression.sigmoid(comb[0][i]))
            if proba[i] <= LOGISTIC_REGRESSION_THRESHOLD:
                new_prediction[0][i] = 0
            else:
                new_prediction[0][i] = 1

            if new_prediction[0][i] == out_params[0][i]:
                accuracy_count += 1

        result_prediction = {'prediction': new_prediction, 'accuracy': (accuracy_count / in_params.shape[1]) * 100,
                             'proba': np.array(proba)}

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
        :return: Для каждой выборки: прогноз для выборки, точность этого прогноза, вероятности, функция ошибок модели.
        """
        w = np.zeros([in_train.shape[0], 1])

        w, dw, costs = MyLogisticRegression.gradient_descent(w, in_train, out_train, num_iterations, learning_rate)

        train_logistic_regression = MyLogisticRegression.prediction(w, in_train, out_train)
        test_logistic_regression = MyLogisticRegression.prediction(w, in_test, out_test)

        return train_logistic_regression, test_logistic_regression


class DataNormalization:
    """
    Нормализация данных.
    """

    @staticmethod
    def minimax_normalize(not_norm_data):
        """
        Минимакс нормализация.
        :param not_norm_data: Ненормализованные данные.
        :return: Нормализованные данные.
        """
        min_vals = np.min(not_norm_data)
        max_vals = np.max(not_norm_data)

        return (not_norm_data - min_vals) / (max_vals - min_vals)

    @staticmethod
    def standardization(not_norm_data):
        """
        Нормализация методом стандартизации.
        :param not_norm_data: Ненормализованные данные.
        :return: Нормализованные данные.
        """
        mean = np.mean(not_norm_data)
        std = np.std(not_norm_data)

        return (not_norm_data - mean) / std


class DataSample:
    """
    Выборка данных.
    """

    def __init__(self, name_cvs):
        """
        Выборка данных.
        """
        self.data = pd.read_csv(name_cvs, header=0, encoding='windows-1251', sep=';', decimal=',')
        """Данные из csv файла."""

        self.cols = list(self.data.columns)
        """Столбцы документа."""

    def convert_to_analyse(self, positive_cols, int_cols):
        """
        Преобразует данные для анализа.
        :param positive_cols: Список столбцов для преобразования в положительные значения.
        :param int_cols: Список столбцов для преобразования в целочисленные значения.
        :return: Данные без пустых строк, с положительным значениями указанных столбцов, и целочисленными значениями
        указанных столбцов.
        """

        self.data.dropna()

        for i in positive_cols:
            self.data = self.data.loc[self.data[self.cols[i]] >= 0]

        for i in int_cols:
            self.data[self.cols[i]] = self.data[self.cols[i]].astype(int)

    def get_train_test_samples(self, client_id=COL_INDEX_NUM_CLIENT, frac_train_sample=LEARNING_PERCENT):
        """
        Получает выборки для обучения и тестирования.
        :return: Датафреймы для обучения и тестирования.
        """

        train_data = self.data.sample(frac=frac_train_sample, random_state=0).sort_values(
            by=[self.cols[client_id]])

        test_data = self.data.drop(train_data.index).sort_values(by=[self.cols[client_id]])

        return train_data, test_data

    @staticmethod
    def get_input_output_data(dataframe, input_col, normalization_func=DataNormalization.standardization,
                              output_col=COL_INDEX_DEBT):
        """
        Получает выходные и выходные данные.
        :param dataframe: Выборка.
        :param normalization_func: Способ нормализации данных.
        :param input_col: Индекс столбца входных данных.
        :param output_col: Индекс столбца выходных данных.
        :return: Входные и выходные данные.
        """
        input_data = dataframe.iloc[:, input_col].values.T
        input_data = normalization_func(input_data)

        output_data = dataframe[dataframe.columns[output_col]].to_numpy().reshape(-1, 1).T

        return input_data, output_data

    @staticmethod
    def generate_new_data(file_name, train_dataframe, result_train, train_proba, test_dataframe, result_test,
                          test_proba):
        """
        Генерирует csv файл.
        :param file_name: Имя файла.
        :param train_dataframe: Датафрейм тренировочных данных.
        :param result_train: Результаты прогнозирования на тренировочных данных.
        :param train_proba: Вероятности модели для тренировочной выборки.
        :param test_dataframe: Датафрейм тестовых данных.
        :param result_test: Результаты прогнозирования тестовых данных.
        :param test_proba: Вероятности модели для тестовой выборки.
        """

        train_dataframe[NAME_PREDICT_COLUMN] = result_train[0].astype(int)
        test_dataframe[NAME_PREDICT_COLUMN] = result_test[0].astype(int)

        train_dataframe[NAME_MODEL_PROBABILITIES_COLUMN] = train_proba
        test_dataframe[NAME_MODEL_PROBABILITIES_COLUMN] = test_proba

        new_dataframe = pd.concat([train_dataframe, test_dataframe])

        new_dataframe = new_dataframe.sort_values(by=[new_dataframe.columns[COL_INDEX_NUM_CLIENT]])
        new_dataframe = new_dataframe.drop(columns=new_dataframe.columns[COL_INDEX_NUM_CLIENT])

        new_dataframe.drop_duplicates(subset=[new_dataframe.columns[COL_INDEX_NUM_CLIENT]], keep='first', inplace=True)

        new_dataframe.to_csv(file_name, sep=';', encoding='windows-1251')


class SamplingStrategy:
    """Стратегия корректировки выборки."""

    @staticmethod
    def oversampling(dataframe, col_attr=COL_INDEX_DEBT, random_coef=0.7):
        """
        Дублирование примеров миноритарного класса.
        :param dataframe: Датафрейм.
        :param col_attr: Индекс столбца, по которому будет проходить корректировка.
        :param random_coef: Коэффициент для случайной выборки миноритарного класса.
        :return: Датафрейм со сбалансированными классами.
        """
        classes_counts = dataframe[dataframe.columns[col_attr]].value_counts()
        minor_class = classes_counts.idxmin()
        major_class = classes_counts.idxmax()

        minor_rows = dataframe.loc[dataframe[dataframe.columns[col_attr]] == minor_class]

        while classes_counts[minor_class] < classes_counts[major_class]:
            dataframe = dataframe.append(minor_rows.sample(int(minor_rows.shape[0] * random_coef)))
            classes_counts = dataframe[dataframe.columns[col_attr]].value_counts()

        return dataframe


class ModelMertics:
    """Метрики модели."""

    def __init__(self, true_output, proba, step=0.01):
        """
        Метрики модели.
        :param true_output: Настоящие классы.
        :param proba: Вероятности модели.
        :param step: Шаг для построения ROC-кривой.
        """
        self.proba = proba
        """Вероятности модели."""
        self.true_output = true_output
        """Настоящие классы."""
        self.step = step
        """Шаг для построения ROC-кривой."""

    def get_confusion_matrix(self, threshold=LOGISTIC_REGRESSION_THRESHOLD):
        """
        Вычисляет матрицу ошибок 1 и 2 рода.
        :param threshold: Пороговое значение.
        :return: Матрица ошибок {true_negative, false_positive, false_negative, true_positive}.
        """
        true_positive = np.sum(np.logical_and(self.proba >= threshold, self.true_output == 1))
        true_negative = np.sum(np.logical_and(self.proba < threshold, self.true_output == 0))
        false_negative = np.sum(np.logical_and(self.proba < threshold, self.true_output == 1))
        false_positive = np.sum(np.logical_and(self.proba >= threshold, self.true_output == 0))

        confusion_matrix = np.array([true_negative, false_positive, false_negative, true_positive]).reshape((2, 2))

        return confusion_matrix

    def get_roc(self):
        """
        Вычисляет точки для ROC-кривой и AUC.
        :return: Массивы точек ROC кривой, AUC.
        """
        thresholds = np.arange(0, 1 + self.step, self.step)
        tpr = []
        fpr = []

        for i in thresholds:
            error_matrix = self.get_confusion_matrix(i)
            tn, fp, fn, tp = error_matrix.flatten()
            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))

        auc = abs(np.trapz(tpr, x=fpr))

        return tpr, fpr, auc, thresholds

    def show_roc(self):
        """
        Показывает график ROC-кривой.
        :return: График ROC-кривой.
        """
        tpr, fpr, auc, t = self.get_roc()
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC-кривая, AUC = {0:0.2f}'.format(auc))
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривая')
        plt.legend(loc="lower right")
        plt.show()

    def get_accuracy(self, threshold=LOGISTIC_REGRESSION_THRESHOLD):
        """
        Точность: отношение количества верных прогнозов к количеству всех прогнозов.
        :param threshold: Граница отсечения модели.
        :return: Точность.
        """
        matrix = self.get_confusion_matrix(threshold)

        return (matrix[1][1] + matrix[0][0]) / (matrix[1][1] + matrix[0][0] + matrix[1][0] + matrix[0][1])

    def get_precision(self, threshold=LOGISTIC_REGRESSION_THRESHOLD):
        """
        Точность: доля прогнозируемых положительных результатов, которые являются действительно верно-положительными
        результатами для всех положительно предсказанных объектов.
        :param threshold: Граница отсечения модели.
        :return: Точность.
        """
        matrix = self.get_confusion_matrix(threshold)

        return matrix[1][1] / (matrix[1][1] + matrix[0][1])

    def get_recall(self, threshold=LOGISTIC_REGRESSION_THRESHOLD):
        """
        Полнота: пропорция всех верно-положительно предсказанных объектов к общему количеству
        действительно положительных.
        :param threshold: Граница отсечения модели.
        :return: Полнота.
        """
        matrix = self.get_confusion_matrix(threshold)

        return matrix[1][1] / (matrix[1][1] + matrix[1][0])

    def get_f_measure(self, threshold=LOGISTIC_REGRESSION_THRESHOLD):
        """
        F-мера: взвешенное гармоническое среднее полноты и точности. Демонстрирует,
        как много случаев прогнозируется моделью правильно, и сколько истинных экземпляров модель не пропустит.
        :param threshold: Граница отсечения модели.
        :return: f-мера.
        """
        precision = self.get_precision(threshold)
        recall = self.get_recall(threshold)

        return 2 * precision * recall / (precision + recall)

    def get_all_mertrics(self, threshold=LOGISTIC_REGRESSION_THRESHOLD):
        """
        Получает все метрики модели для заданного порогового значения модели.
        :param threshold: Пороговое значение.
        :return: Строковое представление метрик модели.
        """
        matrix = self.get_confusion_matrix(threshold)
        accuracy = self.get_accuracy(threshold)
        precision = self.get_precision(threshold)
        recall = self.get_recall(threshold)
        f_measure = self.get_f_measure(threshold)
        tpr, fpr, auc, th = self.get_roc()
        return "Точность (accuracy) = {0:0.2f}. Точность (precision) = {1:0.2f}.\n".format(accuracy, precision) + \
               "Полнота = {0:0.2f}. F-мера = {1:0.2f}.\n".format(recall, f_measure) + \
               "AUC-ROC = {0:0.2f}. Матрица ошибок \n{1}".format(auc, matrix)


class BankData:
    """Поиск границы отсечения."""

    def __init__(self, tpr, fpr, thresholds_model):
        """
        Поиск границы отсечения балла.
        :param tpr: True Positive Rate.
        :param fpr: False Positive Rate.
        :param thresholds_model Пороговые значения модели.
        """
        self.tpr = tpr
        self.fpr = fpr
        self.thresholds = thresholds_model

    def get_optimal(self):
        """
        Вычисляет оптимальную точку ROC-кривой и пороговое значение модели.
        :return: Точка ROC-кривой, оптимальное пороговое значение модели.
        """
        roc_curve = np.column_stack((self.fpr, self.tpr))
        distances = np.sqrt(np.sum(np.square(roc_curve - [0, 1]), axis=1))

        optimal_point = roc_curve[np.argmin(distances)]
        optimal_threshold = self.thresholds[np.argmin(distances)]

        return optimal_point, optimal_threshold


if __name__ == '__main__':
    d = DataSample(NAME_CSV)

    d.convert_to_analyse(int_cols=[COL_INDEX_SCORE], positive_cols=[COL_INDEX_BKI_SCORE])
    data_train, data_test = d.get_train_test_samples()

    data_train = SamplingStrategy.oversampling(data_train)

    input_train, output_train = DataSample.get_input_output_data(data_train, input_col=[COL_INDEX_SCORE])

    input_test, output_test = DataSample.get_input_output_data(data_test, input_col=[COL_INDEX_SCORE])

    print('Обучающие входные и выходные')
    print(input_train, output_train)

    print('Тестовые входные и выходные')
    print(input_test, output_test)

    # region Логистическая регрессия MyLogisticRegression

    train_mlr, test_mlr = MyLogisticRegression.logistic_regression(input_train, output_train, input_test, output_test)

    print('(MyLogisticRegression) Точность на обучающей выборке ' + str(train_mlr['accuracy']) + ' %')
    print('(MyLogisticRegression) Точность на тестовой выборке ' + str(test_mlr['accuracy']) + ' %')

    log_reg_metrics = ModelMertics(output_test, test_mlr['proba'])
    print('Метрики логистической регрессии')
    print(log_reg_metrics.get_all_mertrics())

    log_reg_metrics.show_roc()

    # Создание нового файла
    d.generate_new_data('mlr_data.csv', data_train, train_mlr['prediction'], train_mlr['proba'],
                        data_test, test_mlr['prediction'], test_mlr['proba'])

    # endregion
    tpr_lr, fpr_lr, auc_lr, thresholds_lr = log_reg_metrics.get_roc()

    bd = BankData(tpr_lr, fpr_lr, thresholds_lr)
    print(bd.get_optimal())
