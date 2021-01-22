from datetime import datetime


class TrieStruct:
    def __init__(self, level=0, sup_count=0, max_length=1):
        self.next_level = {}
        self.level = level
        self.sup_count = sup_count
        self.max_length = max_length

    def binary_search(self, key: any):
        next_lvl_list = list(self.next_level)
        nxt_len = len(next_lvl_list)
        if nxt_len == 0:
            return None
        st = 0
        ed = nxt_len - 1
        while True:
            if st == ed or st + 1 == ed:
                if next_lvl_list[st] == key or next_lvl_list[ed] == key:
                    return self.next_level[key]
                return None
            mid = (st + ed) // 2
            if key <= next_lvl_list[mid]:
                ed = mid
            else:
                st = mid

    def update_max_length(self):
        next_level_max_length = [self.next_level[k].max_length for k in self.next_level]
        next_level_max_length.append(self.level)
        self.max_length = max(next_level_max_length)


def print_level_data(level: int, candidates: int, freq_patterns: int):
    print(
        "|"
        + "|".join(
            [
                f"{level}".rjust(6),
                f"{candidates}".rjust(11),
                f"{freq_patterns}|".rjust(16),
            ]
        )
    )
    print("------------------------------------")


def get_one_itemset(data: list, min_sup: int):
    one_item_set = {}
    for transaction in data:
        for elem in transaction:
            try:
                one_item_set[elem] += 1
            except KeyError:
                one_item_set[elem] = 1
    candidates = len(one_item_set)
    temp_struct = list(one_item_set)
    for k in temp_struct:
        if one_item_set[k] < min_sup:
            del one_item_set[k]
    return one_item_set, candidates


def is_exist(root: TrieStruct, lst: list, index: int, ed: int):
    if index == ed:
        return True
    next_node = root.binary_search(lst[index])
    if next_node:
        return is_exist(next_node, lst, index + 1, ed)
    return False


def subset_prune(main_root: TrieStruct):
    pattern_len = len(stack)
    for i in range(pattern_len - 2):
        temp_list = stack[:i] + stack[i + 1 :]
        if not is_exist(main_root, temp_list, 0, pattern_len - 1):
            return True
    return False


def generate_candidates(root: TrieStruct, level: int, main_root: TrieStruct):
    pruned = 0
    if root.level == level - 2:
        next_level_list = list(root.next_level)
        next_level_number = len(next_level_list)
        candidates = (next_level_number * (next_level_number - 1)) // 2

        for i in range(next_level_number):
            stack.append(next_level_list[i])
            for j in range(i + 1, next_level_number):
                stack.append(next_level_list[j])
                if not subset_prune(main_root):
                    new_node = TrieStruct(level, max_length=level)
                    v_node = root.next_level[next_level_list[i]]
                    v_node.next_level[next_level_list[j]] = new_node

                    root.max_length = level
                    v_node.max_length = level

                else:
                    pruned += 1

                stack.pop()

            stack.pop()

        return candidates, pruned
    candidates = 0
    for k, v in root.next_level.items():
        if v.max_length == level - 1:
            stack.append(k)
            sub_candidates, sub_pruned = generate_candidates(v, level, main_root)
            root.update_max_length()
            stack.pop()

            pruned += sub_pruned
            candidates += sub_candidates
    return candidates, pruned


def calculate_support_count(
    root: TrieStruct, transaction: tuple, st: int, ed: int, level: int
):
    if root.level == level:
        root.sup_count += 1
        return

    if st == ed:
        return

    for i in range(st, ed):
        next_node = root.binary_search(transaction[i])
        if next_node and next_node.max_length == level:
            calculate_support_count(next_node, transaction, i + 1, ed, level)


def prune_infrequent(root: TrieStruct, level: int, min_sup: int):
    if root.level == level:
        if root.sup_count < min_sup:
            del root
            return 1
        return 0
    pruned = 0
    next_lvl_list = list(root.next_level)
    for k in next_lvl_list:
        sub_pruned = prune_infrequent(root.next_level[k], level, min_sup)
        pruned += sub_pruned
        if sub_pruned > 0 and root.level == level - 1:
            del root.next_level[k]
    root.update_max_length()
    return pruned


def print_child_support(root: TrieStruct, level: int):
    if root.level == level:
        print(root.sup_count)
        return
    for k, v in root.next_level.items():
        print(f"level: {root.next_level[k].level}", k)
        print_child_support(v, level)


def run_apriori(fle: str, support_per: float):
    data = []
    with open(fle, "r") as f:
        for line in f.readlines():
            data.append(tuple(map(int, line.split())))

    print(f"Mining {fle}...")
    print("------------------------------------")
    print("|".join(["| level", " candidates", " freq. patterns|"]))
    print("------------------------------------")
    total_transactions = len(data)
    max_transaction_length = 0
    for transaction in data:
        length_tran = len(transaction)
        if length_tran > max_transaction_length:
            max_transaction_length = length_tran
    min_sup = round((total_transactions * support_per) / 100.0)

    one_item_set, candidates = get_one_itemset(data, min_sup)
    freq_patterns = len(one_item_set)
    print_level_data(1, candidates, freq_patterns)
    root = TrieStruct()
    child_sorted = list(one_item_set)
    child_sorted.sort()
    for k in child_sorted:
        child = TrieStruct(1, one_item_set[k])
        root.next_level[k] = child

    for level in range(2, max_transaction_length + 1):
        candidates, pruned = generate_candidates(root, level, root)
        for transaction in data:
            calculate_support_count(root, transaction, 0, len(transaction), level)
        pruned += prune_infrequent(root, level, min_sup)
        freq_patterns = candidates - pruned
        if freq_patterns > 0:
            print_level_data(level, candidates, freq_patterns)
        else:
            break


if __name__ == "__main__":
    file_list = ["mushroom.dat", "kosarak.dat", "retail.dat", "T10I4D100K.dat"]
    for fle in file_list:
        stack = []
        then = datetime.now()
        run_apriori(fle, 30.0)
        now = datetime.now()
        print("Execution time:", (now - then).total_seconds(), "seconds")
        print()
