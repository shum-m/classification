# Система оценки кредитных заявок
## О системе
Кредитные организации стремятся максимизировать свою прибыль и минимизировать банковские риски. В связи с этим банки обращаются к классификации клиента банка как надежного или неплатежеспособного заемщика. Система представлена в виде приложения с пользовательским интерфейсом.
### Реализация
1. python 3.7
2. pandas 
3. numpy
4. pyqt5
5. matplotlib

### Методы оценки
В приложении реализованы следующие методы оценки кредитоспособности:
* логистическая регрессия,
* дерево решения,
* наивный байесовский классификатор. 
### Ансамбли
Реализованы ансамбли: 
* модельное усреднение, 
* модельное наложение. 
### Метрики
Для оценки моделей рассматриваются метрики:
* accuracy (точность),
* precision (точность),
* recall (полнота),
* F1-мера,
* ROC-кривая и AUC-ROC площадь под ROC кривой.
### Балансировка классов
1. дублирование примеров миноритарного класса,
2. метод расширения выборки меньшинства синтетическими образцами (SMOTE).
### Методы нормализации данных
1. минимакс нормализация,
2. нормализация алгебраической функцией,
3. нормализация обобщенной алгебраической функцией,
4. нормализация обобщенным вариантом сигмоиды,
5. нормализация арктангенсом.

### Методы стандартизации данных
1. стандартизация методом Парето,
2. стандартизация методом Z-преобразования,
3. сасштабирование стабильности переменной,
4. степенное преобразование.

### Данные
Данные о заемщиках представлены в csv файле. Для каждого заемщика могут быть определены следующие признаки: 
1. номер заемщика,
2. идентификатор заемщика,
3. балл БКИ (бюро кредитных историй)
4. скоринговый балл банка, 
5. дефолт (была ли просрочка по кредиту более 60 дней).

### Использование
При прогнозировании кредитоспособности могут использоваться как только балл банка, так и балл БКИ. На основе признаков делается прогноз будет ли дефолт и вероятность его наступления. Для всего набора заемщиков есть возможность вычислить границы отсечения по баллу банка или баллу БКИ, количество "хороших" и "плохих" заемщиков в портфеле для заданной вероятности дефолта.

### Скриншоты
Создание модели

![изображение_2023-08-17_165441678](https://github.com/shum-m/classification/assets/109040710/b003417f-beb0-46ef-8690-426d3fd8c373)

Метрики некоторой модели

![изображение_2023-08-17_165524317](https://github.com/shum-m/classification/assets/109040710/1c05e0a2-c885-4329-9ae4-df2458cb0775)

Границы отсечения некоторой модели

![изображение_2023-08-17_165719328](https://github.com/shum-m/classification/assets/109040710/9ce6dfd4-9e87-4425-8965-34d5697f4d73)
