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
        if not isinstance(data[0][i], str):
            for j in range(row_count):
                value_set.add(data[j][i])
            if len(value_set) > 10:
                discrete_flag[i] = False
    return discrete_flag


def calculate_result(row: list, root: TreeNode, discrete_flag: list):
    if root.decision:
        return root.decision
    if discrete_flag[root.attribute]:
        return calculate_result(row, root.children[row[root.attribute]], discrete_flag)
    elif row[root.attribute] > root.split_point:
        return calculate_result(row, root.children["greater"], discrete_flag)
    else:
        return calculate_result(row, root.children["less"], discrete_flag)


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


def conditional_sort(db: list, subset_db: list, attribute: int):
    subset_db.sort(key=lambda x: db[x][attribute])


def continuous_best_split(db: list, subset_db: list, attribute: int, entropy_db: float):
    attribute_value_set = set()
    for row in subset_db:
        attribute_value_set.add(str(db[row][attribute]))

    if len(attribute_value_set) == 1:
        return (
            calculate_entropy(db, subset_db),
            attribute_value_set.pop(),
            subset_db,
            [],
        )
    cursor_pos = 0
    cursor_value = str(db[subset_db[0]][attribute])
    subset_db_len = len(subset_db)
    best_split_point = None
    best_gain = None
    best_left_split = None
    best_right_split = None
    for i in range(1, subset_db_len):
        if str(db[subset_db[i]][attribute]) != cursor_value:
            split_point = (float(cursor_value) + db[subset_db[i]][attribute]) / 2
            info_d = ((cursor_pos + 1) / subset_db_len) * calculate_entropy(
                db, subset_db[: cursor_pos + 1]
            ) + ((subset_db_len - cursor_pos - 1) / subset_db_len) * calculate_entropy(
                db, subset_db[cursor_pos + 1 :]
            )
            split_info = -((cursor_pos + 1) / subset_db_len) * log2(
                (cursor_pos + 1) / subset_db_len
            ) - ((subset_db_len - cursor_pos - 1) / subset_db_len) * log2(
                (subset_db_len - cursor_pos - 1) / subset_db_len
            )
            gain = (entropy_db - info_d) / split_info
            if not best_gain:
                best_gain = gain
                best_split_point = split_point
                best_left_split = subset_db[: cursor_pos + 1]
                best_right_split = subset_db[cursor_pos + 1 :]

            if gain > best_gain:
                best_gain = gain
                best_split_point = split_point
                best_left_split = subset_db[: cursor_pos + 1]
                best_right_split = subset_db[cursor_pos + 1 :]
            cursor_value = db[subset_db[i]][attribute]
            cursor_pos = i

    return best_gain, best_split_point, best_left_split, best_right_split


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
    best_split_point = None
    best_left_split = None
    best_right_split = None
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
                conditional_sort(db, subset_db, i)
                gain, split_point, left_split, right_split = continuous_best_split(
                    db, subset_db, attribute, entropy_db
                )

                if not best_gain_ratio:
                    best_gain_ratio = gain
                    best_attribute = i
                    best_split_point = split_point
                    best_left_split = deepcopy(left_split)
                    best_right_split = deepcopy(right_split)

                if gain > best_gain_ratio:
                    best_gain_ratio = gain
                    best_attribute = i
                    best_split_point = split_point
                    best_left_split = deepcopy(left_split)
                    best_right_split = deepcopy(right_split)

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
        root.split_point = best_split_point
        left_child = TreeNode()
        root.children["less"] = left_child
        build_decision_tree(
            db,
            left_child,
            discrete_flag,
            best_left_split,
            new_attribute_flag,
            discrete_value_map,
        )
        right_child = TreeNode()
        root.children["greater"] = right_child
        if len(best_right_split) > 0:
            build_decision_tree(
                db,
                right_child,
                discrete_flag,
                best_right_split,
                new_attribute_flag,
                discrete_value_map,
            )
        else:
            right_child.decision = majority_decision


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
