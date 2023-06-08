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


class LogisticRegression:
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
        weights = LogisticRegression.sigmoid(np.dot(w.T, in_params))
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
            comb = LogisticRegression.calc_weights(w, in_params)
            cost = LogisticRegression.calc_cost(m, out_params, comb)

            dw = LogisticRegression.update_weights(m, comb, in_params, out_params)
            w = w - learning_rate * dw

            costs.append(cost)

        return w, dw, costs

    @staticmethod
    def prediction(w, in_params):
        """
        Делает прогноз, вычисляет точность.
        :param w: Веса.
        :param in_params: Входные параметры.
        :return: Прогноз, вероятности.
        """
        m = in_params.shape[1]
        comb = np.dot(w.T, in_params)
        new_prediction = np.zeros((1, m))
        proba = []

        for i in range(m):
            proba.append(LogisticRegression.sigmoid(comb[0][i]))
            if proba[i] <= LOGISTIC_REGRESSION_THRESHOLD:
                new_prediction[0][i] = 0
            else:
                new_prediction[0][i] = 1

        return {'prediction': new_prediction, 'proba': np.array(proba)}

    @staticmethod
    def logistic_regression(in_train, out_train, in_test, learning_rate=LEARNING_RATE,
                            num_iterations=ITERATIONS):
        """
        Вычисления логистической регрессии для обучающей и тестовой выборок.
        :param in_train: Тренировочные входные данные.
        :param out_train: Тренировочные выходные данные.
        :param in_test: Тестовые входные данные.
        :param learning_rate: Коэффициент обучения.
        :param num_iterations: Количество итераций.
        :return: Для каждой выборки: прогноз для выборки, вероятности.
        """
        w = np.zeros([in_train.shape[0], 1])

        w, dw, costs = LogisticRegression.gradient_descent(w, in_train, out_train, num_iterations, learning_rate)

        train_logistic_regression = LogisticRegression.prediction(w, in_train)
        test_logistic_regression = LogisticRegression.prediction(w, in_test)

        return train_logistic_regression, test_logistic_regression


class NaiveBayesClassifier:
    """
    Наивный байесовский классификатор.
    """

    @staticmethod
    def fit(in_train, out_train):
        """
        Находит уникальные классы, среднее значение и дисперсию по входным данным.
        :param in_train: Входные тренировочные данные.
        :param out_train: Выходные тренировочные данные.
        :return: Классы, среднее значение и дисперсия.
        """
        in_train = in_train.T
        out_train = out_train.T.ravel()
        classes = np.unique(out_train)
        n_features = in_train.shape[1]
        mean = np.zeros((len(classes), n_features))
        var = np.zeros((len(classes), n_features))
        for i, c in enumerate(classes):
            x_class = in_train[c == out_train]
            mean[i, :] = x_class.mean(axis=0)
            var[i, :] = x_class.var(axis=0)
        return classes, mean, var

    @staticmethod
    def predict(in_test, classes, mean, var):
        """
        Делает прогноз.
        :param in_test: Входные данные.
        :param classes: Классы.
        :param mean: Среднее значение.
        :param var: Дисперсия.
        :return: Прогноз.
        """
        in_test = in_test.T
        y_pred = []
        proba = []
        for x in in_test:
            posteriors = []
            for i, c in enumerate(classes):
                prior = np.log(1 / len(classes))
                class_conditional = np.sum(np.log(NaiveBayesClassifier.gaussian_pdf(x, mean[i, :], var[i, :])))
                posterior = prior + class_conditional
                posteriors.append(posterior)

            normalized_posteriors = np.exp(posteriors) / np.sum(np.exp(posteriors))
            proba.append(normalized_posteriors)
            y_pred.append(classes[np.argmax(normalized_posteriors)])
        return np.array(y_pred), np.array(proba)

    @staticmethod
    def gaussian_pdf(x, mean, var):
        """
        Гауссовское распределение.
        :param x: Входные параметры.
        :param mean: Среднее значение.
        :param var: Дисперсия.
        :return: Вероятность признака по гауссовскому распределению.
        """
        return 1 / np.sqrt(2 * np.pi * var) * np.exp(-((x - mean) ** 2) / (2 * var))

    @staticmethod
    def naive_bayes_classification(in_train, out_train, in_test):
        """
        Прогноз наивного байесовского классификатора для набора данных.
        :param in_train: Входные тренировочные данные.
        :param out_train: Выходные тренировочные данные.
        :param in_test: Входные тестовые данные.
        :return: Для каждой выборки: прогноз для выборки, вероятности.
        """
        classes, mean, var = NaiveBayesClassifier.fit(in_train, out_train)

        train_prediction, train_proba = NaiveBayesClassifier.predict(in_train, classes, mean, var)
        test_prediction, test_proba = NaiveBayesClassifier.predict(in_test, classes, mean, var)

        return {"prediction": np.reshape(train_prediction, (1, -1)), "proba": train_proba[:, 1]}, \
               {"prediction": np.reshape(test_prediction, (1, -1)), "proba": test_proba[:, 1]}


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
        min_vals = np.min(not_norm_data, axis=1)[:, np.newaxis]
        max_vals = np.max(not_norm_data, axis=1)[:, np.newaxis]

        return (not_norm_data - min_vals) / (max_vals - min_vals)

    @staticmethod
    def algebraic_function(not_norm_data, a=1):
        """
        Нормализация алгебраической функцией.
        :param not_norm_data: Ненормализованные данные.
        :param a: Коэффициент в нормализации.
        :return: Нормализованные данные.
        """
        return not_norm_data / (np.abs(not_norm_data) + a)

    @staticmethod
    def generalization_algebraic_function(not_norm_data, a=1, n=2):
        """
        Нормализация обобщенной алгебраической функцией.
        :param not_norm_data: Ненормализованные данные.
        :param a: Коэффициент в нормализации.
        :param n: Степень в нормализации.
        :return: Нормализованные данные.
        """
        return np.power(not_norm_data, n) / (np.abs(np.power(not_norm_data), n) + a)

    @staticmethod
    def generalization_sigmoid(not_norm_data, a=1, b=1):
        """
        Нормализация обобщенным вариантом сигмоиды.
        :param not_norm_data: Ненормализованные данные.
        :param a: Коэффициент в нормализации.
        :param b: Коэффициент в нормализации.
        :return: Нормализованные данные.
        """
        return (1 + np.exp(-b * not_norm_data)) ** -a

    @staticmethod
    def arctangent_normalization(not_norm_data, a=1):
        """
        Нормализация арктангенсом.
        :param not_norm_data:
        :param a: Коэффициент нормализации.
        :return: Нормализованные данные.
        """
        return np.arctan(not_norm_data / a)


class DataStandardization:
    """
    Стандартизация данных.
    """

    @staticmethod
    def pareto_method(not_norm_data):
        """
        Стандартизация методом Парето.
        :param not_norm_data: Нестандартизированные данные.
        :return: Стандартизированные данные.
        """
        mean = np.mean(not_norm_data, axis=1)[:, np.newaxis]
        std = np.std(not_norm_data, axis=1)[:, np.newaxis]  # стандартное отклонение
        return (not_norm_data - mean) / std

    @staticmethod
    def z_method(not_norm_data):
        """
        Стандартизация методом Z-преобразования.
        :param not_norm_data: Нестандартизированные данные.
        :return: Стандартизированные данные.
        """
        mean = np.mean(not_norm_data, axis=1)[:, np.newaxis]
        std = np.std(not_norm_data, axis=1)[:, np.newaxis]

        return (not_norm_data - mean) / (std ** 2)

    @staticmethod
    def variable_stability_scaling(not_norm_data):
        """
        Масштабирование стабильности переменной.
        :param not_norm_data: Нестандартизированные данные.
        :return: Стандартизированные данные.
        """
        mean = np.mean(not_norm_data, axis=1)[:, np.newaxis]
        std = np.std(not_norm_data, axis=1)[:, np.newaxis]

        return mean * (not_norm_data - mean) / (std ** 2)

    @staticmethod
    def power_transformation(not_norm_data):
        """
        Степенное преобразование.
        :param not_norm_data: Нестандартизированные данные.
        :return: Стандартизированные данные.
        """
        mean = np.mean(not_norm_data, axis=1)[:, np.newaxis]
        p = np.sqrt(not_norm_data - np.min(not_norm_data, axis=1)[:, np.newaxis])
        return p - (mean ** p)


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
    def get_input_output_data(dataframe, input_col, standartization_func=DataStandardization.pareto_method,
                              normalization_func=None, output_col=COL_INDEX_DEBT,
                              normalization_first=False):
        """
        Получает выходные и выходные данные.
        :param dataframe: Выборка.
        :param input_col: Индекс столбца входных данных.
        :param standartization_func: Способ стандартизации данных.
        :param normalization_func: Способ нормализации данных.
        :param output_col: Индекс столбца выходных данных.
        :param normalization_first: Проводить ли сначала нормализацию если заданы способы и стандартизации,
        и нормализации.
        :return: Входные и выходные данные.
        """

        input_data = dataframe.iloc[:, input_col].values.T

        if normalization_func is not None and normalization_first is True:
            input_data = normalization_func(input_data)
            if standartization_func is not None:
                input_data = standartization_func(input_data)
        else:
            if standartization_func is not None:
                input_data = standartization_func(input_data)
                if normalization_func is not None:
                    input_data = normalization_func(input_data)

        output_data = dataframe[dataframe.columns[output_col]].to_numpy().reshape(-1, 1).T

        return input_data, output_data

    @staticmethod
    def generate_new_data(train_dataframe, result_train, train_proba, test_dataframe, result_test,
                          test_proba):
        """
        Генерирует новый датафрейм из разделенных на выборки.
        :param train_dataframe: Датафрейм тренировочных данных.
        :param result_train: Результаты прогнозирования на тренировочных данных.
        :param train_proba: Вероятности модели для тренировочной выборки.
        :param test_dataframe: Датафрейм тестовых данных.
        :param result_test: Результаты прогнозирования тестовых данных.
        :param test_proba: Вероятности модели для тестовой выборки.
        :return: Объединенный датафрейм.
        """

        train_dataframe[NAME_PREDICT_COLUMN] = result_train[0].astype(int)
        test_dataframe[NAME_PREDICT_COLUMN] = result_test[0].astype(int)

        train_dataframe[NAME_MODEL_PROBABILITIES_COLUMN] = train_proba
        test_dataframe[NAME_MODEL_PROBABILITIES_COLUMN] = test_proba

        new_dataframe = pd.concat([train_dataframe, test_dataframe])

        new_dataframe = new_dataframe.sort_values(by=[new_dataframe.columns[COL_INDEX_NUM_CLIENT]])
        new_dataframe = new_dataframe.drop(columns=new_dataframe.columns[COL_INDEX_NUM_CLIENT])

        new_dataframe.drop_duplicates(subset=[new_dataframe.columns[COL_INDEX_NUM_CLIENT]], keep='first', inplace=True)

        return new_dataframe

    @staticmethod
    def write_csv(dataframe, filename):
        """
        Записывает датафрейм в файл.
        :param dataframe: Датафрейм.
        :param filename: Имя файла.
        """
        dataframe.to_csv(filename, sep=';', encoding='windows-1251')


class SamplingStrategy:
    """Стратегия корректировки выборки."""

    @staticmethod
    def get_classes(dataframe, col_attr=COL_INDEX_DEBT):
        """
        Получает миноритарный и мажоритарный класс в столбце датафрейма.
        :param dataframe: Датафрем.
        :param col_attr: Индекс столбца, для получения классов.
        :return: Массив с метками и количествами, метки мажоритарного и миноритарного классов.
        """
        classes_counts = dataframe[dataframe.columns[col_attr]].value_counts()
        minor_class = classes_counts.idxmin()
        major_class = classes_counts.idxmax()
        return classes_counts, major_class, minor_class

    @staticmethod
    def oversampling(dataframe, col_attr=COL_INDEX_DEBT, random_coef=0.7):
        """
        Дублирование примеров миноритарного класса.
        :param dataframe: Датафрейм.
        :param col_attr: Индекс столбца, по которому будет проходить корректировка.
        :param random_coef: Коэффициент для случайной выборки миноритарного класса.
        :return: Датафрейм со сбалансированными классами.
        """
        classes_counts, major_class, minor_class = SamplingStrategy.get_classes(dataframe, col_attr)

        minor_rows = dataframe.loc[dataframe[dataframe.columns[col_attr]] == minor_class]

        while classes_counts[minor_class] < classes_counts[major_class]:
            dataframe = dataframe.append(minor_rows.sample(int(minor_rows.shape[0] * random_coef)))
            classes_counts = dataframe[dataframe.columns[col_attr]].value_counts()

        return dataframe

    @staticmethod
    def smote(dataframe, k=3, col_attr=COL_INDEX_DEBT, oversampling_coef=10, selected_columns=None):
        """
        Добавляет синтетические примеры миноритарного класса.
        :param dataframe: Датафрейм.
        :param k: Количество ближайших соседей для генерации синтетических примеров.
        :param col_attr: Индекс столбца, по которому будет проходить корректировка.
        :param oversampling_coef: Коэффициент количества искусственных примеров.
        :param selected_columns: Список индексов столбцов, для которых генерируются примеры.
        :return: Датафрейм со сбалансированными классами.
        """
        classes_counts, major_class, minor_class = SamplingStrategy.get_classes(dataframe, col_attr)

        minor_dataframe = dataframe[dataframe.iloc[:, col_attr] == minor_class]
        minor_samples = minor_dataframe.iloc[:, selected_columns].values.astype(float)
        size_minor_sample = minor_samples.shape[0]

        synthetic_samples = []
        size_synthetic_samples = size_minor_sample * oversampling_coef

        for i in range(size_synthetic_samples):
            rand_sample = minor_samples[np.random.randint(0, size_minor_sample)]

            distances = np.sqrt(np.sum((minor_samples - rand_sample) ** 2, axis=1))
            sorted_indices = np.argsort(distances)
            k_nearest_neighbors = sorted_indices[1:k + 1]

            random_neighbor_idx = np.random.randint(0, k)
            neighbor = minor_samples[k_nearest_neighbors[random_neighbor_idx]]

            alpha = np.random.uniform(0, 1)
            synthetic_sample = rand_sample + alpha * (neighbor - rand_sample)

            synthetic_samples.append(synthetic_sample)

        synthetic_df = pd.DataFrame(np.array(synthetic_samples), columns=dataframe.columns[selected_columns])
        synthetic_df[dataframe.columns[col_attr]] = minor_class

        balanced_df = pd.concat([dataframe, synthetic_df], ignore_index=True)

        return balanced_df


class ModelMetrics:
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
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC-кривая, Площадь под кривой (AUC) = {0:0.2f}'.format(auc))
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Доля ложно положительных прогнозов')
        plt.ylabel('Доля истинно положительных прогнозов')
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

    def get_all_metrics(self, threshold=LOGISTIC_REGRESSION_THRESHOLD):
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


class DefaultModel:
    """
    Модель построения и отсечения заявок для заданного уровня дефолта.
    """

    def __init__(self, dataframe):
        """
        Модель построения и отсечения заявок для заданного уровня дефолта.
        :param dataframe: Датафрейм.
        """
        self.dataframe = dataframe

    def count_default_orders(self, col_default_proba=NAME_MODEL_PROBABILITIES_COLUMN, default_rate=0.05):
        """
        Подсчитывает количество дефолтных заявок для заданного уровня дефолтности.
        :param default_rate: Уровень дефолта.
        :param col_default_proba: Наименование столбца по которому считаются заявки.
        :return: Количество дефолтных и не дефолтных заявок, их процентное соотношение к общему числу заявок.
        """
        default_count = self.dataframe[self.dataframe[col_default_proba].astype(float) <= default_rate].shape[0]
        good_count = self.dataframe[self.dataframe[col_default_proba].astype(float) > default_rate].shape[0]

        default_percent = default_count / (default_count + good_count) * 100

        return default_count, good_count, default_percent, 100 - default_percent

    def form_clipping_board(self, col_index_bank_scoring=COL_INDEX_SCORE, col_index_bki_scoring=COL_INDEX_BKI_SCORE,
                            col_default_proba=NAME_MODEL_PROBABILITIES_COLUMN, default_rate=0.05):
        """
        Формирует границы отсечения для банковского балла и балла БКИ при заданном уровне дефолтности.
        :param col_index_bki_scoring: Индекс столбца баллов БКИ.
        :param col_index_bank_scoring: Индекс столбца скорингового балла банка.
        :param default_rate: Уровень дефолта.
        :param col_default_proba: Наименование столбца по которому считаются заявки.
        :return: Граница скорингового балла банка, граница скорингового балла БКИ.
        """
        self.dataframe[col_default_proba] = self.dataframe[col_default_proba].astype(float)

        col_bank_scoring = self.dataframe.columns[col_index_bank_scoring]
        col_bki_scoring = self.dataframe.columns[col_index_bki_scoring]

        try:
            id_board = self.dataframe[self.dataframe[col_default_proba] <= default_rate][
                col_default_proba].idxmax()
        except ValueError:
            return 0, 0

        bank_board = self.dataframe.loc[id_board, col_bank_scoring]
        bki_board = self.dataframe.loc[id_board, col_bki_scoring]

        return bank_board, bki_board


class BootstrapSampler:
    """Бутстрэп выборка."""

    def __init__(self, in_data, out_data):
        """
        Бутстрэп выборка.
        :param in_data: Входные данные.
        :param out_data: Выходные данные.
        """
        self.input_data = in_data
        self.output_data = out_data

    def generate_sample(self, samples_number):
        """
        Генерирует бутстрэп выборку.
        :param samples_number: Количество выборок.
        :return: Бутстрэп выборки для входных и выходных данных.
        """
        columns = self.input_data.shape[1]

        column_indices = np.random.choice(columns, size=(samples_number, columns), replace=True)
        in_train_split = [self.input_data[:, cols] for cols in column_indices]
        out_train_split = [self.output_data[:, cols] for cols in column_indices]

        return in_train_split, out_train_split


class Ensemble:
    """Ансамбль моделей."""

    def __init__(self, *models):
        """
        Ансамбль моделей.
        :param args: Модели для создания ансамбля.
        """
        self.models = models


class EnsembleModelAverage(Ensemble):
    """Ансамбль методом модельного усреднения."""

    def use(self, in_train, out_train, in_test, voting_threshold=0.5):
        """
        Использовать ансамбль.
        :param in_train: Тренировочные входные данные.
        :param out_train: Тренировочные выходные данные.
        :param in_test: Тестовые входные данные.
        :param voting_threshold: Порог для голосования в бинарной классификации.
        :return: Прогноз на основе ансамбля.
        """
        in_train_split, out_train_split = BootstrapSampler(in_train, out_train).generate_sample(len(self.models))

        test_models_predictions = []
        test_models_proba = []

        for i in range(len(self.models)):
            prediction = self.models[i](in_train_split[i], out_train_split[i], in_test)
            test_models_predictions.append(prediction[1]['prediction'])
            test_models_proba.append(prediction[1]['proba'])

        ensemble_test_prediction_probs = np.mean(np.array(test_models_proba), axis=0)
        ensemble_test_prediction_classes = np.where(ensemble_test_prediction_probs >= voting_threshold, 1, 0) \
            .reshape(1, -1)

        return {'prediction': ensemble_test_prediction_classes, 'proba': ensemble_test_prediction_probs}


class StackingEnsemble:
    """Ансамбль методом модельного наложения."""

    def __init__(self, meta_model, *models):
        """
        Ансамбль методом модельного наложения.
        :param meta_model: Мета-модель.
        :param models: Базовые модели.
        """
        self.models = models
        self.meta_model = meta_model

    def use(self, in_train, out_train, in_test):
        train_predictions = np.zeros((len(self.models), in_train.shape[1]))
        test_predictions = np.zeros((len(self.models), in_test.shape[1]))

        # Обучение базовых моделей и формирование стекинговых данных для тренировочного набора
        for i in range(len(self.models)):
            train_predictions[i, :] = self.models[i](in_train, out_train, in_train)[1]['prediction']

        # Обучение базовых моделей и формирование стекинговых данных для тестового набора
        for i in range(len(self.models)):
            test_predictions[i, :] = self.models[i](in_train, out_train, in_test)[1]['prediction']

        print('ПОСЧИТАНО тренировочные: ', train_predictions.shape)
        print('ПОСЧИТАНО тестовые: ', test_predictions.shape)

        return self.meta_model(train_predictions, out_train, test_predictions)[1]


class DecisionTree:
    """Решающее дерево."""

    def __init__(self):
        self.tree = None

    @staticmethod
    def gini_index(groups):
        """
        Вычисляет индекс Джини.
        :param groups: Группы разделения.
        :return: Индекс Джини
        """
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            p = np.count_nonzero(group) / size
            gini += (p * (1 - p))
        return gini

    @staticmethod
    def split(index, value, dataset):
        """
        Разделяет данные на две группы.
        :param index: Индекс элемента.
        :param value: Порог разделения
        :param dataset: Набор данных.
        :return: Группы.
        """
        left = dataset[:, dataset[index] < value]
        right = dataset[:, dataset[index] >= value]
        return left, right

    @staticmethod
    def get_best_split(dataset):
        """
        Поиск наилучшего разделения.
        :param dataset: Набор данных для разделения
        :return: Индекс признака для лучшего разделения, значение для разделения, пара групп для разделения.
        """
        b_index, b_value, b_score, b_groups = np.inf, np.inf, np.inf, None
        for index in range(len(dataset) - 1):
            for value in np.unique(dataset[index]):
                groups = DecisionTree.split(index, value, dataset)
                gini = DecisionTree.gini_index(groups)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, value, gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    @staticmethod
    def to_terminal(group):
        """
        Делает прогноз для листового узла.
        :param group: Группы разделения.
        :return: Прогноз для листового узла.
        """
        outcomes, counts = np.unique(group, return_counts=True)
        return outcomes[np.argmax(counts)]

    @staticmethod
    def split_recursive(node, max_depth, min_size, depth):
        """
        Разделяет дерево.
        :param node: Текущий узел дерева.
        :param max_depth: Максимальная глубина.
        :param min_size: Минимальное количество объектов для листа.
        :param depth: Глубина.
        :return: Разделенное дерево.
        """

        left, right = node['groups']
        del node['groups']
        if not left.any() or not right.any():
            node['left'] = node['right'] = DecisionTree.to_terminal(np.concatenate((left, right), axis=1))
            return
        if depth >= max_depth:
            node['left'], node['right'] = DecisionTree.to_terminal(left[-1]), DecisionTree.to_terminal(right[-1])
            return
        if left.shape[1] <= min_size:
            node['left'] = DecisionTree.to_terminal(left[-1])
        else:
            node['left'] = DecisionTree.get_best_split(left)
            DecisionTree.split_recursive(node['left'], max_depth, min_size, depth + 1)
        if right.shape[1] <= min_size:
            node['right'] = DecisionTree.to_terminal(right[-1])
        else:
            node['right'] = DecisionTree.get_best_split(right)
            DecisionTree.split_recursive(node['right'], max_depth, min_size, depth + 1)

    def fit(self, in_train, out_train, max_depth=5, min_size=5):
        """
        Обучение модели.
        :param in_train: Входные тренировочные данные.
        :param out_train: Выходные тренировочные данные.
        :param max_depth: Глубина дерева.
        :param min_size: Минимальное количество для листа.
        """
        dataset = np.concatenate((in_train, out_train), axis=0)
        self.tree = DecisionTree.get_best_split(dataset)
        DecisionTree.split_recursive(self.tree, max_depth, min_size, 1)

    @staticmethod
    def predict_recursive(node, row):
        """
        Рекурсивное прогнозирование.
        :param node: Текущий узел дерева.
        :param row: Данные для прогноза.
        :return: Прогноз.
        """
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return DecisionTree.predict_recursive(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return DecisionTree.predict_recursive(node['right'], row)
            else:
                return node['right']

    def predict(self, in_test):
        """
        Прогноз для тестовых данных.
        :param in_test: Тестовые входные данные.
        :return: Прогноз для тестовых данных.
        """
        predictions = []
        for i in range(in_test.shape[1]):
            row = in_test[:, i]
            prediction = DecisionTree.predict_recursive(self.tree, row)
            predictions.append(prediction)
        return np.array(predictions).reshape(1, -1)

    @staticmethod
    def decision_tree(in_train, out_train, in_test):
        """
        Прогнозы для выборок.
        :param in_train: Входные тренировочные данные.
        :param out_train: Выходные тренировочные данные.
        :param in_test: Входные тестовые данные.
        :return: Прогноз для каждой выборки, вероятности для каждой выборки.
        """
        dt = DecisionTree()
        dt.fit(in_train, out_train)
        train_prediction = dt.predict(in_train)
        test_prediction = dt.predict(in_test)

        return {'prediction': train_prediction, 'proba': train_prediction.reshape(-1)}, \
               {'prediction': test_prediction, 'proba': test_prediction.reshape(-1)}


if __name__ == '__main__':
    d = DataSample(NAME_CSV)

    d.convert_to_analyse(int_cols=[COL_INDEX_SCORE], positive_cols=[COL_INDEX_BKI_SCORE])
    data_train, data_test = d.get_train_test_samples()

    data_train = SamplingStrategy.oversampling(data_train)
    # data_train = SamplingStrategy.smote(data_train, selected_columns=[COL_INDEX_SCORE])

    input_train, output_train = DataSample.get_input_output_data(data_train,
                                                                 input_col=[COL_INDEX_SCORE])

    input_test, output_test = DataSample.get_input_output_data(data_test,
                                                               input_col=[COL_INDEX_SCORE])

    print('Обучающие входные и выходные')
    print(input_train, output_train)

    print('Тестовые входные и выходные')
    print(input_test, output_test)

    # region Метод решающих деревьев.

    dt_train, dt_test = DecisionTree.decision_tree(input_train, output_train, input_test)

    dt_metrics = ModelMetrics(output_test, dt_test['proba'])
    print(dt_test['proba'].shape)
    print(dt_test['prediction'].shape)
    print('Метрики решающего дерева')
    print(dt_metrics.get_all_metrics())

    # Создание нового файла
    d.write_csv(d.generate_new_data(data_train, dt_train['prediction'], dt_train['proba'], data_test,
                                    dt_test['prediction'], dt_test['proba']), 'dt_data.csv')

    # endregion

    # region МЕТОД МОДЕЛЬНОГО НАЛОЖЕНИЯ
    se = StackingEnsemble(NaiveBayesClassifier.naive_bayes_classification,
                          LogisticRegression.logistic_regression,
                          NaiveBayesClassifier.naive_bayes_classification,
                          DecisionTree.decision_tree)

    se_test = se.use(input_train, output_train, input_test)

    se_metrics = ModelMetrics(output_test, se_test['proba'])
    print('Метрики ансамбля методом модельного наложения (тест)')
    print(se_metrics.get_all_metrics())
    # endregion

    # region Ансамбли МЕТОД МОДЕЛЬНОГО УСРЕДНЕНИЯ

    # ema = EnsembleModelAverage(
    #     NaiveBayesClassifier.naive_bayes_classification,
    #     NaiveBayesClassifier.naive_bayes_classification,
    #     NaiveBayesClassifier.naive_bayes_classification,
    #     NaiveBayesClassifier.naive_bayes_classification,
    #     LogisticRegression.logistic_regression,
    #     LogisticRegression.logistic_regression,
    #     LogisticRegression.logistic_regression
    # )
    #
    # sens_test = ema.use(input_train, output_train, input_test)
    # sens_metrics = ModelMetrics(output_test, sens_test['proba'])
    # print('Метрики ансамбля методом модельного усреднения (тест)')
    # print(sens_metrics.get_all_metrics())

    # endregion

    # region Ансамбли МЕТОД МОДЕЛЬНОГО УСРЕДНЕНИЯ

    ema = EnsembleModelAverage(
        DecisionTree.decision_tree,
        DecisionTree.decision_tree,
        DecisionTree.decision_tree,
        NaiveBayesClassifier.naive_bayes_classification,
        LogisticRegression.logistic_regression
    )

    sens_test = ema.use(input_train, output_train, input_test)
    sens_metrics = ModelMetrics(output_test, sens_test['proba'])
    print('Метрики ансамбля методом модельного усреднения (тест)')
    print(sens_metrics.get_all_metrics())

    # endregion

    # region Логистическая регрессия LogisticRegression

    train_mlr, test_mlr = LogisticRegression.logistic_regression(input_train, output_train, input_test)
    print(test_mlr['prediction'].shape)
    log_reg_metrics = ModelMetrics(output_test, test_mlr['proba'])
    print('Метрики логистической регрессии')
    print(log_reg_metrics.get_all_metrics())

    # log_reg_metrics.show_roc()

    # Создание нового файла
    # d.write_csv(d.generate_new_data(data_train, train_mlr['prediction'], train_mlr['proba'], data_test,
    #                                test_mlr['prediction'], test_mlr['proba']), 'mlr_data.csv')

    # endregion

    # region Наивный байесовский классификатор
    train_nbc, test_nbc = NaiveBayesClassifier.naive_bayes_classification(input_train, output_train, input_test)

    nbc_metrics = ModelMetrics(output_test, test_nbc["proba"])
    print('Метрики наивного байесовского классификатора')
    print(nbc_metrics.get_all_metrics())

    # nbc_metrics.show_roc()

    # Создание нового файла
    # d.write_csv(d.generate_new_data(data_train, train_nbc['prediction'], train_nbc['proba'],
    #                                data_test, test_nbc['prediction'], test_nbc['proba']), 'nbc_data.csv')
    # endregion

    # region Логистическая регрессия границы и количество заявок
    print('ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ')
    boards = DefaultModel(DataSample('mlr_data.csv').data)

    print('Количество и доли заявок')
    mlr_default, mlr_good, mlr_default_rate, mlr_good_rate = boards.count_default_orders(default_rate=0.05)
    print("Дефолтные: ", mlr_default, " Кредитоспособные: ", mlr_good, "  Доля дефолтных в портфеле: ",
          mlr_default_rate, " Доля кредитоспособных в портфеле: ", mlr_good_rate)

    mlr_bank_board, mlr_bki_board = boards.form_clipping_board(default_rate=0.05)
    print("Граница по БКИ: ", mlr_bki_board, "Граница по скоринговому баллу: ", mlr_bank_board)
    # endregion

    # region Наивный байесовский классификатор границы и количество заявок
    print('НАИВНЫЙ БАЙЕСОВСКИЙ КЛАССИФИКАТОР')
    boards = DefaultModel(DataSample('nbc_data.csv').data)

    print('Количество и доли заявок')
    nbc_default, nbc_good, nbc_default_rate, nbc_good_rate = boards.count_default_orders(default_rate=0.05)
    print("Дефолтные: ", nbc_default, " Кредитоспособные: ", nbc_good, " Доля дефолтных в портфеле: ", nbc_default_rate,
          " Доля кредитоспособных в портфеле: ", nbc_good_rate)

    nbc_bank_board, nbc_bki_board = boards.form_clipping_board(default_rate=0.05)
    print("Граница по БКИ: ", nbc_bki_board, "Граница по скоринговому баллу: ", nbc_bank_board)
    # endregion
