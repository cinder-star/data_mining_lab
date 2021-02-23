import random
import operator
from math import sqrt, pi, exp
from sklearn.metrics import classification_report


def read_file(filename: str):
    file = open(filename, "r")
    data = []
    for line in file.readlines():
        row = list(map(str, line.split()))
        data.append(row)
    return data


def process_data(data: list):
    processed_data = []
    for row in data:
        processed_row = []
        for attribute in row:
            try:
                if "." in attribute:
                    processed_row.append(float(attribute))
                else:
                    processed_row.append(int(attribute))
            except ValueError:
                processed_row.append(attribute)
        processed_data.append(processed_row)

    return processed_data


def validation_split(data: list, percentile: float):
    random.shuffle(data)
    split_point = round(percentile * len(data))
    train_db = data[:split_point]
    test_db = data[split_point:]
    return train_db, test_db


def class_separation(db: list):
    separate_by_class = {}
    for row in db:
        try:
            separate_by_class[row[-1]].append(row)
        except KeyError:
            separate_by_class[row[-1]] = [row]
    return separate_by_class


def calculate_mean(values: list):
    return sum(values) / len(values)


def calculate_standard_devaition(values: list):
    mean = calculate_mean(values)
    varience = sum([(x - mean) ** 2 for x in values]) / (len(values) - 1)
    return sqrt(varience)


def db_summarize(db: list[list]):
    summary = [
        (calculate_mean(column), calculate_standard_devaition(column), len(column))
        for column in zip(*db)
    ]
    del summary[-1]
    return summary


def class_summarize(db: list[list]):
    separate_by_class = class_separation(db)
    class_summary = {}
    for k, v in separate_by_class.items():
        class_summary[k] = db_summarize(v)
    return class_summary


def calculate_probability(x: float, avg: float, standard_dev: float):
    try:
        exponent = exp(-((x - avg) ** 2 / (2 * standard_dev ** 2)))
        temp = (1 / (sqrt(2 * pi) * standard_dev)) * exponent
    except ZeroDivisionError:
        temp = 0
    return temp


def calculate_prediction_probabilities(summary: any, row: list):
    total_rows = sum([summary[cls_label][0][2] for cls_label in summary])
    probabilities = {}
    for k, v in summary.items():
        probabilities[k] = summary[k][0][2] / float(total_rows)
        summary_len = len(v)
        for i in range(summary_len):
            mean, stdev, _ = v[i]
            probabilities[k] *= calculate_probability(row[i], mean, stdev)
    return probabilities


def convert_to_number(string: str):
    return int.from_bytes(string.encode(), "little")


def argmax(map: dict):
    return max(map.items(), key=operator.itemgetter(1))[0]


def prepare_data(data: list):
    prepared_data = []
    map = {}
    count = 0
    for row in data:
        for i, elem in enumerate(row):
            if isinstance(elem, str):
                if elem in map:
                    row[i] = map[elem]
                else:
                    map[elem] = count
                    row[i] = count
                    count += 1
        prepared_data.append(row)
    return prepared_data


if __name__ == "__main__":
    file = "iris.data"
    data = read_file(file)
    data = process_data(data)
    data = prepare_data(data)
    train_db, test_db = validation_split(data, 0.7)
    db_summary = db_summarize(data)
    class_summary = class_summarize(train_db)
    real_label = [row[-1] for row in test_db]
    prediction = []
    classes = list(class_summary.keys())
    for row in test_db:
        probabilities = calculate_prediction_probabilities(class_summary, row)
        prediction.append(argmax(probabilities))
    f = open("bayesoutput.txt", "a+")
    f.write(f"{file}\n{classification_report(real_label, prediction, labels=classes)}")
