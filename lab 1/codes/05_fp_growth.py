from datetime import datetime
import operator as op
from functools import reduce
from memory_profiler import memory_usage

class Trie:
    def __init__(self, parent=None, sup_count=0, value=None):
        self.children = {}
        self.parent = parent
        self.sup_count = sup_count
        self.value = value


def ncr(n, r):
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def read_from_file(file: str):
    data = []
    with open(file, "r") as f:
        for line in f.readlines():
            data.append((tuple(map(int, line.split())), 1))
    return data


def get_one_length_hashmap(data: list, min_sup: int):
    temp_map = {}
    for transaction, count in data:
        for elem in transaction:
            try:
                temp_map[elem] += count
            except KeyError:
                temp_map[elem] = count

    temp_struct = list(temp_map)
    for k in temp_struct:
        if temp_map[k] < min_sup:
            del temp_map[k]

    temp_list = [(v, k) for k, v in temp_map.items()]
    temp_list.sort(reverse=True)

    return {v: {"sup_count": k, "addresses": []} for k, v in temp_list}


def preprocess_data(data: list, cmp_map: dict):
    temp_db = []
    for transaction, count in data:
        processed_transaction = [
            (cmp_map[elem]["sup_count"], elem)
            for elem in transaction
            if elem in cmp_map.keys()
        ]
        if len(processed_transaction) > 0:
            temp_db.append((processed_transaction, count))
    for transaction, _ in temp_db:
        transaction.sort(reverse=True)
    processed_data = []
    for transaction, count in temp_db:
        processed_data.append((tuple(elem[1] for elem in transaction), count))
    return processed_data


def reverse_path(child: Trie):
    if child.parent.parent == None:
        return (child.value,)
    return (*reverse_path(child.parent), child.value)


def build_tree(
    root: Trie, transaction: tuple, count: int, hash_map: dict, index: int, end: int
):
    if index == end:
        return
    try:
        root.children[transaction[index]].sup_count += count
    except KeyError:
        child = Trie(root, count, transaction[index])
        root.children[transaction[index]] = child
        hash_map[transaction[index]]["addresses"].append(child)
    finally:
        build_tree(
            root.children[transaction[index]],
            transaction,
            count,
            hash_map,
            index + 1,
            end,
        )


def traverse_tree(root: Trie, stack: list):
    if len(root.children) == 0:
        print(stack)
        return
    for k, v in root.children.items():
        stack.append((k, v.sup_count))
        traverse_tree(v, stack)
        stack.pop()


def run_fp_growth(data: list, min_sup: int):
    if len(data) == 1:
        trans_len = len(data[0][0])
        if data[0][1] < min_sup:
            return 0
        return 2 ** trans_len - 1 - trans_len
    hash_map = get_one_length_hashmap(data, min_sup)
    data = preprocess_data(data, hash_map)
    root = Trie()
    for transaction, count in data:
        build_tree(root, transaction, count, hash_map, 0, len(transaction))
    keys = list(hash_map)
    key_len = len(keys) - 1
    count = 0
    for i in range(key_len, 0, -1):
        mini_tree_transactions = []
        total_transaction = []
        for address_link in hash_map[keys[i]]["addresses"]:
            if address_link.parent.value:
                transaction = (
                    reverse_path(address_link.parent),
                    address_link.sup_count,
                )
                if len(transaction[0]) > 1:
                    mini_tree_transactions.append(transaction)
                total_transaction.append(transaction)
        mini_hashmap = get_one_length_hashmap(total_transaction, min_sup)
        run_ans = run_fp_growth(mini_tree_transactions, min_sup)
        hash_len = len(mini_hashmap)
        count = count + hash_len + run_ans

    return count


def main_func():
    file_list = ["chess.dat"]
    for file in file_list:
        data = read_from_file(file)
        min_sup = round((len(data) * 90.0) / 100.0)
        then = datetime.now()
        pattern_count = len(get_one_length_hashmap(data, min_sup)) + run_fp_growth(
            data, min_sup
        )
        now = datetime.now()
        print("freqent patterns:", pattern_count)
        print("Execution time:", (now - then).total_seconds(), "seconds")


if __name__ == "__main__":
    mem = max(memory_usage(proc=main_func))
    print(f"Peak memory usage: {mem} MiB")