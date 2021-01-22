class Trie:
    def __init__(self, parent=None, sup_count=0):
        self.children = {}
        self.parent = parent
        self.sup_count = sup_count


def read_from_file(file: str):
    data = []
    with open(file, "r") as f:
        for line in f.readlines():
            data.append(tuple(map(int, line.split())))
    return data


def get_one_length_hashmap(data: list):
    temp_map = {}
    for transaction in data:
        for elem in transaction:
            try:
                temp_map[elem] += 1
            except KeyError:
                temp_map[elem] = 1
    temp_list = [(v, k) for k, v in temp_map.items()]
    temp_list.sort(reverse=True)

    return {v: {"sup_count": k, "addresses": []} for k, v in temp_list}


def preprocess_data(data: list, cmp_map: dict):
    temp_db = []
    for transaction in data:
        temp_db.append([(cmp_map[elem]["sup_count"], elem) for elem in transaction])
    for transaction in temp_db:
        transaction.sort(reverse=True)
    processed_data = []
    for transaction in temp_db:
        processed_data.append(tuple(elem[1] for elem in transaction))
    return processed_data


def build_tree(root: Trie, transaction: tuple, hash_map: dict, index: int, end: int):
    if index == end:
        return
    try:
        root.children[transaction[index]].sup_count += 1
    except KeyError:
        child = Trie(root, 1)
        root.children[transaction[index]] = child
        hash_map[transaction[index]]["addresses"].append(child)
    finally:
        build_tree(
            root.children[transaction[index]], transaction, hash_map, index + 1, end
        )


def traverse_tree(root: Trie, stack: list):
    if len(root.children) == 0:
        print(stack)
        return
    for k, v in root.children.items():
        stack.append(k)
        traverse_tree(v, stack)
        stack.pop()


def run_fp_growth(data: list, hash_map: dict):
    root = Trie()
    for transaction in data[:2]:
        build_tree(root, transaction, hash_map, 0, len(transaction))


if __name__ == "__main__":
    file_list = ["mushroom.dat"]
    for file in file_list:
        data = read_from_file(file)
        hashmap = get_one_length_hashmap(data)
        data = preprocess_data(data, hashmap)
        run_fp_growth(data, hashmap)
