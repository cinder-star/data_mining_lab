import random
import operator
from copy import deepcopy
from math import log2


class TreeNode:
    def __init__(self, attribute=None, value=None, split_point=None, decision=None):
        self.children = {}
        self.attribute = attribute
        self.value = value
        self.split_point = split_point
        self.decision = decision


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


def classify_attributes(data: list):
    row_count = len(data)
    attribute_count = len(data[0][:-1])
    discrete_flag = [True for _ in range(attribute_count)]
    for i in range(attribute_count):
        value_set = set()
        if isinstance(data[0][i], str):
            continue
        else:
            for j in range(row_count):
                value_set.add(data[j][i])
            if len(value_set) > 10:
                discrete_flag[i] = False
    return discrete_flag


def calculate_result(row: list, root: TreeNode):
    if root.decision:
        return root.decision
    return calculate_result(row, root.children[row[root.attribute]])


def validation_split(data: list, percentile: float):
    random.shuffle(data)
    split_point = round(percentile * len(data))
    train_db = data[:split_point]
    test_db = data[split_point:]
    return train_db, test_db


def majority_decision(data: list, subset_db: list):
    decision_map = {}
    for index in subset_db:
        try:
            decision_map[data[index][-1]] += 1
        except KeyError:
            decision_map[data[index][-1]] = 1
    return max(decision_map.items(), key=operator.itemgetter(1))[0]


def calculate_entropy(db: list, subset_db: list):
    total = len(subset_db)
    item_map = {}
    entropy = 0
    for index in subset_db:
        try:
            item_map[db[index][-1]] += 1
        except KeyError:
            item_map[db[index][-1]] = 1
    if len(list(item_map.keys())) == 1:
        return 0
    for k in item_map:
        rat = item_map[k] / total
        entropy -= rat * log2(rat)
    return entropy


def calculate_split_info(db: list, attribute: int):
    count_att = {}
    for row in db:
        try:
            count_att[row[attribute]] += 1
        except KeyError:
            count_att[row[attribute]] = 1
    db_len = len(db)
    split_info = 0
    for k in count_att:
        rat = count_att[k] / db_len
        split_info -= rat * log2(rat)
    return split_info


def calculate_discrete_entropy(db: list, subset_db: list, attribute: int):
    unique_value_map = {}
    for row in subset_db:
        try:
            unique_value_map[db[row][attribute]].append(row)
        except KeyError:
            unique_value_map[db[row][attribute]] = [row]
    info_d = 0
    db_len = len(subset_db)
    for _, v in unique_value_map.items():
        split_len = len(v)
        info_d += (split_len / db_len) * calculate_entropy(db, v)
    return info_d, unique_value_map


def discrete_attribute_value_map(data: list, discrete_flag: list):
    attribute_value_map = {}
    for i, attribute in enumerate(discrete_flag):
        if attribute:
            att_set = set()
            for row in data:
                att_set.add(row[i])
        attribute_value_map[i] = list(att_set)
    return attribute_value_map


def build_decision_tree(
    db: list,
    root: TreeNode,
    discrete_flag: list,
    subset_db: list,
    attribute_flag: list,
    discrete_value_map: dict,
):
    majority_label = majority_decision(db, subset_db)
    entropy_db = calculate_entropy(db, subset_db)
    if entropy_db == 0:
        root.decision = db[subset_db[0]][-1]
        return

    if attribute_flag[0] and len(set(attribute_flag)) == 1:
        root.decision = majority_decision(db, subset_db)
        return

    new_attribute_flag = deepcopy(attribute_flag)
    best_attribute = None
    best_gain_ratio = None
    best_split_info = None
    for i, attribute in enumerate(attribute_flag):
        if not attribute:
            if discrete_flag[i]:
                split_info = calculate_split_info(db, i)
                info_d, split_db = calculate_discrete_entropy(db, subset_db, i)
                ratio = (entropy_db - info_d) / split_info

                if not best_gain_ratio:
                    best_gain_ratio = ratio
                    best_attribute = i
                    best_split_info = split_db

                if ratio > best_gain_ratio:
                    best_gain_ratio = ratio
                    best_attribute = i
                    best_split_info = split_db
            else:
                pass
    root.attribute = best_attribute
    new_attribute_flag[best_attribute] = True
    if discrete_flag[best_attribute]:
        for k in discrete_value_map[best_attribute]:
            if k not in best_split_info:
                child = TreeNode(decision=majority_label)
                root.children[k] = child
            else:
                child = TreeNode()
                root.children[k] = child
                build_decision_tree(
                    db,
                    child,
                    discrete_flag,
                    best_split_info[k],
                    new_attribute_flag,
                    discrete_value_map,
                )
    else:
        pass


if __name__ == "__main__":
    data = read_file("iris.data")
    data = process_data(data)
    discrete_flag = classify_attributes(data)
    discrete_value_map = discrete_attribute_value_map(data, discrete_flag)
    train_db, test_db = validation_split(data, 0.7)
    train_length = len(train_db)
    root = TreeNode()
    build_decision_tree(
        train_db,
        root,
        discrete_flag,
        [i for i in range(train_length)],
        [False for _ in discrete_flag],
        discrete_value_map,
    )
    validation_result = []
    for row in test_db:
        validation_result.append((calculate_result(row, root), row[-1]))
    print(validation_result)
