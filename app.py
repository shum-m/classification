from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QMutex
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QDialog

import border_model_design
import create_model_design
import main_design
import model_metrics_design
import create_ensemble_design
from main import DataSample, LogisticRegression, NaiveBayesClassifier, DecisionTree, DataStandardization, \
    DataNormalization, SamplingStrategy, ModelMetrics, DefaultModel, EnsembleModelAverage, StackingEnsemble

ENSEMBLE_MODEL_AVERAGE_TAB_INDEX = 0
"""Индекс окна ансамбля модельного усреднения."""

STACKING_ENSEMBLE_MODEL_TAB_INDEX = 1
"""Индекс окна ансамбля модельного наложения."""


class LoadCsvThread(QtCore.QThread):
    """
    Загрузка CSV файла.
    """
    load_finished = QtCore.pyqtSignal(DataSample)

    def __init__(self, filename):
        """
        Загрузка CSV файла.
        :param filename: Путь к файлу.
        """
        super().__init__()
        self.filename = filename

    def run(self):
        """
        Запуск потока.
        """
        df = DataSample(self.filename)
        self.load_finished.emit(df)


class CreationModel(QtCore.QThread):
    """
    Формирование модели.
    """
    creation_finished = QtCore.pyqtSignal(object, object, object, object, object, object)

    def __init__(self, dataframe, model_type, standardization, normalization, is_normalization_first, balance_type,
                 train_data_rate, input_data, output_data):
        """
        Инициализация модели.
        :param dataframe: Датафрейм.
        :param model_type: Тип модели.
        :param standardization: Функция стандартизации.
        :param normalization: Функция нормализации.
        :param is_normalization_first: Выполняется ли сначала нормализация.
        :param balance_type: Тип балансировки.
        :param train_data_rate: Доля обучающей выборки.
        :param input_data: Индексы столбцов входных данных.
        :param output_data: Индекс столбца выходных данных.
        """
        super().__init__()
        self.data = dataframe
        self.model_type = model_type
        self.standardization = standardization
        self.normalization = normalization
        self.is_normalization_first = is_normalization_first
        self.balance_type = balance_type
        self.train_data_rate = train_data_rate
        self.input_data = input_data
        self.output_data = output_data

        self.train_data_model = None
        self.test_data_model = None

    def run(self):
        # Предобработка данных
        self.data.convert_to_analyse(int_cols=self.input_data, positive_cols=self.input_data)
        data_train, data_test = self.data.get_train_test_samples(client_id=min(self.input_data),
                                                                 frac_train_sample=self.train_data_rate / 100.0)
        if self.balance_type is not None:
            data_train = self.balance_type(data_train)

        input_train, output_train = DataSample.get_input_output_data(dataframe=data_train, input_col=self.input_data,
                                                                     standartization_func=self.standardization,
                                                                     normalization_func=self.normalization,
                                                                     output_col=self.output_data,
                                                                     normalization_first=self.is_normalization_first)

        input_test, output_test = DataSample.get_input_output_data(dataframe=data_test, input_col=self.input_data,
                                                                   standartization_func=self.standardization,
                                                                   normalization_func=self.normalization,
                                                                   output_col=self.output_data,
                                                                   normalization_first=self.is_normalization_first)

        # Обучение модели
        self.train_data_model, self.test_data_model = self.model_type(input_train, output_train, input_test)

        # Метрики модели
        metrics = ModelMetrics(output_test, self.test_data_model['proba'])

        # Границы отсечения
        borders = DefaultModel(DataSample.generate_new_data(data_train, self.train_data_model['prediction'],
                                                            self.train_data_model['proba'], data_test,
                                                            self.test_data_model['prediction'],
                                                            self.test_data_model['proba']))

        self.creation_finished.emit(data_train, data_test, self.train_data_model, self.test_data_model, metrics,
                                    borders)


class CreationEnsemble(QtCore.QThread):
    creation_finished = QtCore.pyqtSignal(object, object, object, object, object, object)

    def __init__(self, dataframe, ensemble_type, meta_model, lr_count, nbc_count, dt_count,
                 standardization, normalization, is_normalization_first,
                 balance_type, train_data_rate, input_data, output_data):
        """
        Инициализация модели.
        :param dataframe: Датафрейм.
        :param ensemble_type: Тип ансамбля.
        :param meta_model: Мета-модель.
        :param lr_count: Количество моделей логистической регрессии.
        :param nbc_count: Количество моделей наивного байесовского классификатора.
        :param dt_count: Количество моделей решающих деревьев.
        :param standardization: Функция стандартизации.
        :param normalization: Функция нормализации.
        :param is_normalization_first: Выполняется ли сначала нормализация.
        :param balance_type: Тип балансировки.
        :param train_data_rate: Доля обучающей выборки.
        :param input_data: Индексы столбцов входных данных.
        :param output_data: Индекс столбца выходных данных.
        """
        super().__init__()
        self.data = dataframe
        self.ensemble_type = ensemble_type
        self.meta_model = meta_model
        self.lr_count = lr_count
        self.nbc_count = nbc_count
        self.dt_count = dt_count
        self.standardization = standardization
        self.normalization = normalization
        self.is_normalization_first = is_normalization_first
        self.balance_type = balance_type
        self.train_data_rate = train_data_rate
        self.input_data = input_data
        self.output_data = output_data

        self.train_data_model = None
        self.test_data_model = None
        self.mutex = QMutex()

    def run(self):
        self.mutex.lock()
        # Предобработка данных
        self.data.convert_to_analyse(int_cols=self.input_data, positive_cols=self.input_data)
        data_train, data_test = self.data.get_train_test_samples(client_id=min(self.input_data),
                                                                 frac_train_sample=self.train_data_rate / 100.0)

        if self.balance_type is not None:
            data_train = self.balance_type(data_train)

        input_train, output_train = DataSample.get_input_output_data(dataframe=data_train, input_col=self.input_data,
                                                                     standartization_func=self.standardization,
                                                                     normalization_func=self.normalization,
                                                                     output_col=self.output_data,
                                                                     normalization_first=self.is_normalization_first)

        input_test, output_test = DataSample.get_input_output_data(dataframe=data_test, input_col=self.input_data,
                                                                   standartization_func=self.standardization,
                                                                   normalization_func=self.normalization,
                                                                   output_col=self.output_data,
                                                                   normalization_first=self.is_normalization_first)
        # Создание ансамбля
        base_models = [LogisticRegression.logistic_regression] * self.lr_count + \
                      [NaiveBayesClassifier.naive_bayes_classification] * self.nbc_count + \
                      [DecisionTree.decision_tree] * self.dt_count

        if self.ensemble_type is EnsembleModelAverage:
            self.ensemble_type = self.ensemble_type(*base_models)
        elif self.ensemble_type is StackingEnsemble:
            self.ensemble_type = self.ensemble_type(self.meta_model, *base_models)

        self.train_data_model = self.ensemble_type.use(input_train, output_train, input_train)

        self.test_data_model = self.ensemble_type.use(input_train, output_train, input_test)

        # Метрики модели
        metrics = ModelMetrics(output_test, self.test_data_model['proba'])

        # Границы отсечения
        borders = DefaultModel(DataSample.generate_new_data(data_train, self.train_data_model['prediction'],
                                                            self.train_data_model['proba'], data_test,
                                                            self.test_data_model['prediction'],
                                                            self.test_data_model['proba']))
        self.mutex.unlock()
        self.creation_finished.emit(data_train, data_test, self.train_data_model, self.test_data_model, metrics,
                                    borders)


class SaveModelResult(QtCore.QThread):
    """
    Сохранение CSV файла.
    """

    def __init__(self, filename, train_dataframe, result_train, train_proba, test_dataframe, result_test, test_proba):
        """
        Сохранение CSV файла.
        :param filename: Путь к файлу.
        :param train_dataframe: Тренировочный датафрейм.
        :param result_train: Прогноз для тренировочного датафрейма.
        :param train_proba: Вероятности для тренировочного датафрейма.
        :param test_dataframe: Тестовый датафрейм.
        :param result_test: Прогноз для тестового датафрейма.
        :param test_proba:Вероятности для тестового датафрейма.
        """
        super().__init__()
        self.filename = filename
        self.train_dataframe = train_dataframe
        self.result_train = result_train
        self.train_proba = train_proba
        self.test_dataframe = test_dataframe
        self.result_test = result_test
        self.test_proba = test_proba

    def run(self):
        """
        Запуск потока.
        """
        DataSample.write_csv(DataSample.generate_new_data(self.train_dataframe, self.result_train, self.train_proba,
                                                          self.test_dataframe, self.result_test, self.test_proba),
                             self.filename)


class LoadingMessageBox(QMessageBox):
    """
    Сообщение о загрузке.
    """

    def __init__(self, title, message):
        """
        Сообщение о загрузке.
        :param title: Заголовок.
        :param message: Сообщение.
        """
        (super(LoadingMessageBox, self).__init__())
        self.setIcon(QMessageBox.Information)

        self.setWindowTitle(title)
        self.setText(message)
        self.setStandardButtons(QMessageBox.NoButton)


class AnalyticApp(QMainWindow, main_design.Ui_MainWindow):
    """
    Приложение для анализа данных о заемщиках.
    """

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.file_menu.triggered.connect(self.open_file)
        self.create_model.triggered.connect(self.new_model)
        self.create_ensemble.triggered.connect(self.new_ensemble)
        self.model_metrics.triggered.connect(self.show_metrics)
        self.save_model.triggered.connect(self.save_file)
        self.model_borders.triggered.connect(self.get_boards)

        self.loading_message = None

        self.model = None
        """Модель для обучения."""
        self.ensemble = None
        """Ансамбль моделей для обучения."""
        self.data = None
        """Данные для обучения."""
        self.model_file = None
        """Файл для сохранения результатов модели."""

    def open_file(self):
        """
        Выбор файла данных.
        """
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter("Файлы CSV (*.csv)")
        file_dialog.fileSelected.connect(self.load_csv)
        file_dialog.exec_()

    def load_csv(self, filename):
        """
        Загрузка csv файла
        :param filename: Путь к файлу.
        :return: Датафрейм.
        """
        self.loading_message = LoadingMessageBox("Загрузка файла", "Подождите, идет загрузка файла...")
        self.loading_message.show()
        self.set_enabled(False)

        loading_thread = LoadCsvThread(filename)
        loading_thread.load_finished.connect(self.display_csv)
        loading_thread.finished.connect(lambda: self.set_enabled(True))

        loading_thread.start()
        loading_thread.wait()

    def set_enabled(self, enabled):
        """
        Устанавливает доступность элементов.
        :param enabled: Доступность.
        """
        self.table_view.setEnabled(enabled)
        self.file_menu.setEnabled(enabled)
        self.model_menu.setEnabled(enabled)
        self.select_file.setEnabled(enabled)
        self.create_model.setEnabled(enabled)
        self.model_metrics.setEnabled(enabled)
        self.model_borders.setEnabled(enabled)
        self.save_model.setEnabled(enabled)
        if enabled is True:
            self.loading_message.reject()

    @QtCore.pyqtSlot(DataSample)
    def display_csv(self, df):
        """
        Показывает датафрейм.
        :param df: Датафрейм.
        """
        self.data = df
        model = QStandardItemModel(self)

        headers = df.data.columns.tolist()
        model.setColumnCount(len(headers))
        model.setHorizontalHeaderLabels(headers)

        for _, row in df.data.iterrows():
            items = [QtGui.QStandardItem(str(item)) for item in row]
            model.appendRow(items)

        self.table_view.setModel(model)
        self.table_view.show()

    def new_model(self):
        """
        Создает новую модель.
        :return: Новая модель.
        """
        if self.data is not None:
            self.model = ModelCreation(self.data)
            self.ensemble = None
            self.model.show()
            self.model.exec_()
        else:
            QMessageBox.critical(self, "Ошибка", "Нужно выбрать файл данных для создания модели.")

    def new_ensemble(self):
        """
        Создает новый ансамбль моделей.
        :return: Новый ансамбль моделей.
        """
        if self.data is not None:
            self.ensemble = EnsembleCreation(self.data)
            self.model = None
            self.ensemble.show()
            self.ensemble.exec_()
        else:
            QMessageBox.critical(self, "Ошибка", "Нужно выбрать файл данных для создания ансамбля.")

    def show_metrics(self):
        """
        Показывает метрики созданной модели.
        """
        if self.data is not None and ((self.model is not None and self.model.model_created is True)
                                      or (self.ensemble is not None and self.ensemble.ensemble_created is True)):
            metrics_show = MetricsShow(self.model.model_metrics
                                       if self.model is not None
                                       else self.ensemble.ensemble_metrics)
            metrics_show.show()
            metrics_show.exec_()
        else:
            QMessageBox.critical(self, "Ошибка", "Модель не найдена.")

    def save_file(self):
        """
        Сохраняет результаты модели в файл.
        """
        if self.data is not None and ((self.model is not None and self.model.model_created is True)
                                      or (self.ensemble is not None and self.ensemble.ensemble_created is True)):
            options = QFileDialog.Options()

            file_dialog = QFileDialog()
            file_dialog.setOptions(options)
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            file_dialog.setNameFilter("Файлы CSV (*.csv)")

            if file_dialog.exec_() == QFileDialog.Accepted:
                self.model_file = file_dialog.selectedFiles()[0]

                self.loading_message = LoadingMessageBox('Сохранение файла', 'Подождите, файл сохраняется')
                self.loading_message.show()
                self.set_enabled(False)

                if self.model is not None:
                    saving_thread = SaveModelResult(self.model_file, self.model.data_train,
                                                    self.model.train_data_model['prediction'],
                                                    self.model.train_data_model['proba'],
                                                    self.model.data_test, self.model.test_data_model['prediction'],
                                                    self.model.test_data_model['proba'])
                elif self.ensemble is not None:
                    saving_thread = SaveModelResult(self.model_file, self.ensemble.data_train,
                                                    self.ensemble.train_data_ensemble['prediction'],
                                                    self.ensemble.train_data_ensemble['proba'],
                                                    self.ensemble.data_test,
                                                    self.ensemble.test_data_ensemble['prediction'],
                                                    self.ensemble.test_data_ensemble['proba'])

                saving_thread.finished.connect(lambda: self.set_enabled(True))

                saving_thread.start()
                saving_thread.wait()

                self.set_enabled(True)
        else:
            QMessageBox.critical(self, "Ошибка", "Нет модели для сохранения.")

    def get_boards(self):
        """
        Показывает границы отсечения модели.
        """
        if self.data is not None and ((self.model is not None and self.model.model_created is True)
                                      or (self.ensemble is not None and self.ensemble.ensemble_created is True)):

            borders_show = BordersShow(self.model.borders if self.model is not None else self.ensemble.borders)
            borders_show.show()
            borders_show.exec_()
        else:
            QMessageBox.critical(self, "Ошибка", "Модель не найдена.")


class MetricsShow(QDialog, model_metrics_design.Ui_Dialog):
    """
    Окно метрик модели.
    """

    def __init__(self, metrics):
        """
        Задание метрик.
        :param metrics: Метрики.
        """
        (super(MetricsShow, self).__init__())
        self.setupUi(self)

        self.metrics = metrics

        self.accuracy_data.setText(str(round(self.metrics.get_accuracy(), 3)))

        self.precision_data.setText(str(round(self.metrics.get_precision(), 3)))
        self.recall_data.setText(str(round(self.metrics.get_recall(), 3)))
        self.f_measure_data.setText(str(round(self.metrics.get_f_measure(), 3)))

        tpr, fpr, auc, thresholds = self.metrics.get_roc()
        self.auc_roc_data.setText(str(round(auc, 3)))

        confusion_matrix = self.metrics.get_confusion_matrix()

        self.true_positive_data.setText(str(confusion_matrix[1][1]))
        self.true_negative_data.setText(str(confusion_matrix[0][0]))
        self.false_positive_data.setText(str(confusion_matrix[0][1]))
        self.false_negative_data.setText(str(confusion_matrix[1][0]))

        self.show_roc_curve.clicked.connect(lambda: self.metrics.show_roc())


class BordersShow(QDialog, border_model_design.Ui_Dialog):
    """
    Окно границ отсечения модели.
    """
    def __init__(self, borders):
        """
        Задание метрик.
        :param borders: Границы отсечения.
        """
        (super(BordersShow, self).__init__())
        self.setupUi(self)
        self.borders = borders
        self.default_rate = None

        self.calculate_button.clicked.connect(self.get_borders)

    def get_borders(self):
        """
        Вычисляет границы отсечения для заданного уровня дефолтности.
        """

        default_count, good_count, default_percent, good_percent = self.borders.count_default_orders(
            default_rate=self.default_data.value() / 100)

        bank_board, bki_board = self.borders.form_clipping_board(col_index_bank_scoring=self.sckoring_data.value() - 1,
                                                                 col_index_bki_scoring=self.bki_data.value() - 1,
                                                                 default_rate=self.default_data.value() / 100)

        self.default_part_data.setText(str(round(default_percent, 3)) + '%')
        self.good_part_data.setText(str(round(good_percent, 3)) + '%')
        self.default_count_data.setText(str(default_count))
        self.good_count_data.setText(str(good_count))

        self.border_scoring_data.setText(str(bank_board))
        self.border_bki_data.setText(str(bki_board))


class ModelCreation(QDialog, create_model_design.Ui_dialog):
    """
    Создание модели.
    """

    def __init__(self, dataframe):
        """
        Создание модели.
        :param dataframe: Датафрейм.
        """
        super().__init__()
        self.setupUi(self)

        self.create_button.clicked.connect(self.select_params)
        self.loading_message = None

        # Параметры модели.
        self.data = dataframe
        self.model_type = None
        self.standardization = None
        self.normalization = None
        self.is_normalization_first = None
        self.balance_type = None
        self.train_data_rate = None
        self.input_data = None
        self.output_data = None

        self.train_data_model = None
        """Результат работы модели для тренировочных данных."""

        self.test_data_model = None
        """Результат работы модели для тестовых данных."""

        self.model_metrics = None
        """Метрики модели."""

        self.data_train = None
        """Датафрейм тренировочных данных."""

        self.data_test = None
        """Датафрейм тестовых данных."""

        self.borders = None
        """Границы отсечения модели."""

        self.model_created = False
        """Создана ли модель."""

    def select_params(self):
        """
        Создание новой модели по параметрам
        """
        good_params = True

        if self.data_percent.value() == 0:
            QMessageBox.critical(self, "Ошибка", "Не задан размер обучающей выборки.")
            good_params = False
        else:
            self.train_data_rate = self.data_percent.value()

        if self.input_data_text.toPlainText() == "":
            QMessageBox.critical(self, "Ошибка", "Не заданы индексы столбцов входных данных.")
            good_params = False
        else:
            self.input_data = self.input_data_text.toPlainText().split(',')
            self.input_data = [int(i) for i in self.input_data]

        if self.output_data_text.toPlainText() == "":
            QMessageBox.critical(self, "Ошибка", "Не задан индекс столбца выходных данных.")
            good_params = False
        else:
            self.output_data = int(self.output_data_text.toPlainText())

        if good_params is True:
            self.model_created = True

            self.model_type = LearningObjectParameter.get_model_type(self.select_type.currentIndex())
            self.standardization = LearningObjectParameter.get_standardization(self.select_standartization.currentIndex())
            self.normalization = LearningObjectParameter.get_normalization(self.select_normalization.currentIndex())
            self.is_normalization_first = self.normalization_first.isChecked()
            self.balance_type = LearningObjectParameter.get_balance_type(self.select_balance.currentIndex())

            self.loading_message = LoadingMessageBox("Создание модели", "Подождите, идет создание модели...")
            self.loading_message.show()

            self.set_enabled(False)

            creation_thread = CreationModel(self.data, self.model_type, self.standardization, self.normalization,
                                            self.is_normalization_first, self.balance_type, self.train_data_rate,
                                            self.input_data, self.output_data)
            creation_thread.creation_finished.connect(self.set_model_output)
            creation_thread.finished.connect(lambda: self.set_enabled(True))
            creation_thread.start()
            creation_thread.wait()

            QMessageBox.information(self, "Создание модели", "Модель успешно создана.")
            self.close()

    @QtCore.pyqtSlot(object, object, object, object, object, object)
    def set_model_output(self, data_train, data_test, train_data_model, test_data_model, metrics, borders):
        """
        Получение результатов модели.
        :param data_train: Тренировочный датафрейм.
        :param data_test: Тестовый датафрейм.
        :param train_data_model: Результаты на тренировочных данных.
        :param test_data_model: Результаты на тестовых данных.
        :param metrics: Метрики модели.
        :param borders: Границы отсечения модели.
        """
        self.train_data_model = train_data_model
        self.test_data_model = test_data_model
        self.data_train = data_train
        self.data_test = data_test
        self.model_metrics = metrics
        self.borders = borders

    def set_enabled(self, enabled):
        """
        Устанавливает доступность элементов.
        :param enabled: Доступность.
        """
        self.select_type.setEnabled(enabled)
        self.select_normalization.setEnabled(enabled)
        self.select_standartization.setEnabled(enabled)
        self.normalization_first.setEnabled(enabled)
        self.select_balance.setEnabled(enabled)
        self.data_percent.setEnabled(enabled)
        self.input_data_text.setEnabled(enabled)
        self.output_data_text.setEnabled(enabled)
        self.create_button.setEnabled(enabled)
        if enabled is True:
            self.loading_message.reject()


class LearningObjectParameter:
    """
    Параметры обучающегося объекта.
    """
    @staticmethod
    def get_model_type(index):
        """
        Получает модель по индексу.
        :param index: Индекс.
        :return: Модель.
        """
        if index == 0:
            return LogisticRegression.logistic_regression
        elif index == 1:
            return NaiveBayesClassifier.naive_bayes_classification
        elif index == 2:
            return DecisionTree.decision_tree

    @staticmethod
    def get_ensemble_type(index):
        if index == 0:
            return EnsembleModelAverage
        elif index == 1:
            return StackingEnsemble

    @staticmethod
    def get_standardization(index):
        """
        Получает способ стандартизации по индексу.
        :param index: Индекс.
        :return: Функция стандартизация.
        """
        if index == 0:
            return None
        elif index == 1:
            return DataStandardization.pareto_method
        elif index == 2:
            return DataStandardization.z_method
        elif index == 3:
            return DataStandardization.variable_stability_scaling
        elif index == 4:
            return DataStandardization.power_transformation

    @staticmethod
    def get_normalization(index):
        """
        Получает способ нормализации по индексу.
        :param index: Индекс.
        :return: Функция нормализации.
        """
        if index == 0:
            return None
        elif index == 1:
            return DataNormalization.minimax_normalize
        elif index == 2:
            return DataNormalization.algebraic_function
        elif index == 3:
            return DataNormalization.generalization_algebraic_function
        elif index == 4:
            return DataNormalization.generalization_sigmoid
        elif index == 5:
            return DataNormalization.arctangent_normalization

    @staticmethod
    def get_balance_type(index):
        """
        Получает способ балансировки по индексу.
        :param index: Индекс.
        :return: Способ балансировки.
        """
        if index == 0:
            return None
        elif index == 1:
            return SamplingStrategy.oversampling
        elif index == 2:
            return SamplingStrategy.smote


class EnsembleCreation(QDialog, create_ensemble_design.Ui_Dialog):
    """
    Создание ансамбля.
    """
    def __init__(self, dataframe):
        """
        Создание ансамбля.
        :param dataframe: Датафрейм.
        """
        super().__init__()
        self.setupUi(self)

        self.create_button.clicked.connect(self.select_params)
        self.loading_message = None

        # Параметры ансамбля.
        self.data = dataframe
        self.ensemble_type = None
        self.meta_model_type = None
        self.logistic_regression_count = None
        self.naive_bc_count = None
        self.decision_tree_count = None
        self.standardization = None
        self.normalization = None
        self.is_normalization_first = None
        self.balance_type = None
        self.train_data_rate = None
        self.input_data = None
        self.output_data = None

        self.train_data_ensemble = None
        """Результат работы ансамбля для тренировочных данных."""

        self.test_data_ensemble = None
        """Результат работы ансамбля для тестовых данных."""

        self.ensemble_metrics = None
        """Метрики модели."""

        self.data_train = None
        """Датафрейм тренировочных данных."""

        self.data_test = None
        """Датафрейм тестовых данных."""

        self.borders = None
        """Границы отсечения модели."""

        self.ensemble_created = False
        """Создан ли ансамбль."""

    def select_params(self):
        """Создание новой модели по параметрам"""

        good_params = True

        if self.data_percent.value() == 0:
            QMessageBox.critical(self, "Ошибка", "Не задан размер обучающей выборки.")
            good_params = False
        else:
            self.train_data_rate = self.data_percent.value()

        if self.input_data_text.toPlainText() == "":
            QMessageBox.critical(self, "Ошибка", "Не заданы индексы столбцов входных данных.")
            good_params = False
        else:
            self.input_data = self.input_data_text.toPlainText().split(',')
            self.input_data = [int(i) for i in self.input_data]

        if self.output_data_text.toPlainText() == "":
            QMessageBox.critical(self, "Ошибка", "Не задан индекс столбца выходных данных.")
            good_params = False
        else:
            self.output_data = int(self.output_data_text.toPlainText())

        if self.ensembles_widget.currentIndex() == ENSEMBLE_MODEL_AVERAGE_TAB_INDEX and ValidationObject.\
                validate_model_count(self.logistic_regression_count_ema.value(),
                                     self.naive_bc_count_ema.value(),
                                     self.decision_tree_count_ema.value()) is True:
            self.logistic_regression_count = self.logistic_regression_count_ema.value()
            self.naive_bc_count = self.naive_bc_count_ema.value()
            self.decision_tree_count = self.decision_tree_count_ema.value()
        elif self.ensembles_widget.currentIndex() == STACKING_ENSEMBLE_MODEL_TAB_INDEX and ValidationObject.\
                validate_model_count(self.logistic_regression_count_se.value(),
                                     self.naive_bc_count_se.value(),
                                     self.decision_tree_count_se.value()) is True:
            self.meta_model_type = LearningObjectParameter.get_model_type(self.select_meta_model.currentIndex())
            self.logistic_regression_count = self.logistic_regression_count_se.value()
            self.naive_bc_count = self.naive_bc_count_se.value()
            self.decision_tree_count = self.decision_tree_count_se.value()
        else:
            QMessageBox.critical(self, "Ошибка", "Ансамбль должен состоять хотя бы из одной модели.")
            good_params = False

        if good_params is True:
            self.ensemble_created = True

            self.ensemble_type = LearningObjectParameter.get_ensemble_type(self.ensembles_widget.currentIndex())

            self.standardization = LearningObjectParameter.get_standardization(
                self.select_standartization.currentIndex())
            self.normalization = LearningObjectParameter.get_normalization(self.select_normalization.currentIndex())
            self.is_normalization_first = self.normalization_first.isChecked()
            self.balance_type = LearningObjectParameter.get_balance_type(self.select_balance.currentIndex())

            self.set_enabled(False)
            self.loading_message = LoadingMessageBox("Создание ансамбля", "Подождите, идет создание ансамбля...")
            self.loading_message.show()

            creation_thread = CreationEnsemble(self.data, self.ensemble_type, self.meta_model_type,
                                               self.logistic_regression_count, self.naive_bc_count,
                                               self.decision_tree_count, self.standardization, self.normalization,
                                               self.is_normalization_first, self.balance_type, self.train_data_rate,
                                               self.input_data, self.output_data)

            creation_thread.creation_finished.connect(self.set_model_output)
            creation_thread.finished.connect(lambda: self.set_enabled(True))
            creation_thread.start()
            creation_thread.wait()

            QMessageBox.information(self, "Создание ансамбля", "Ансамбль успешно создан.")
            self.close()

    @QtCore.pyqtSlot(object, object, object, object, object, object)
    def set_model_output(self, data_train, data_test, train_data_model, test_data_model, metrics, borders):
        """
        Получение результатов ансамбля.
        :param data_train: Тренировочный датафрейм.
        :param data_test: Тестовый датафрейм.
        :param train_data_model: Результаты на тренировочных данных.
        :param test_data_model: Результаты на тестовых данных.
        :param metrics: Метрики ансамбля.
        :param borders: Границы отсечения ансамбля.
        """
        self.train_data_ensemble = train_data_model
        self.test_data_ensemble = test_data_model
        self.data_train = data_train
        self.data_test = data_test
        self.ensemble_metrics = metrics
        self.borders = borders

    def set_enabled(self, enabled):
        """
        Устанавливает доступность элементов.
        :param enabled: Доступность.
        """
        self.select_normalization.setEnabled(enabled)
        self.select_standartization.setEnabled(enabled)
        self.normalization_first.setEnabled(enabled)
        self.select_balance.setEnabled(enabled)
        self.data_percent.setEnabled(enabled)
        self.input_data_text.setEnabled(enabled)
        self.output_data_text.setEnabled(enabled)
        self.create_button.setEnabled(enabled)

        self.logistic_regression_count_se.setEnabled(enabled)
        self.naive_bc_count_se.setEnabled(enabled)
        self.decision_tree_count_se.setEnabled(enabled)
        self.select_meta_model.setEnabled(enabled)

        self.logistic_regression_count_ema.setEnabled(enabled)
        self.naive_bc_count_ema.setEnabled(enabled)
        self.decision_tree_count_ema.setEnabled(enabled)
        self.ensembles_widget.setEnabled(enabled)

        self.ensemble_model_average.setEnabled(enabled)
        self.stacking_ensemble.setEnabled(enabled)
        self.ensembles_widget.setEnabled(enabled)
        if enabled is True:
            self.loading_message.reject()


class ValidationObject:
    """Проверка параметров объекта обучения."""

    @staticmethod
    def validate_model_count(lr_count, nbc_count, dt_count):
        """Проверяет количество моделей в ансамбле."""
        if lr_count == 0 and nbc_count == 0 and dt_count == 0:
            return False
        else:
            return True


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = AnalyticApp()
    window.show()
    app.exec_()
