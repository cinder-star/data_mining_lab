import random
from copy import deepcopy
from sklearn.metrics import silhouette_score


class Point:
    def __init__(self, attributes: list, cluster: int = -1, distance: float = None):
        self.attributes = attributes
        self.cluster = cluster
        self.distance = distance

    def manhatten_distance(self, point: any):
        distance = 0.0
        for a, b in zip(self.attributes, point.attributes):
            distance += abs(a - b)
        return distance

    def __str__(self):
        return f"{self.attributes} {self.cluster} {self.distance}"

    def __repr__(self):
        return f"{self.attributes} {self.cluster} {self.distance}"


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


def assign_cluster(centroids: list[Point], points: list[Point]):
    changed = 0
    old_clusters = [x.cluster for x in points]
    for point in points:
        for i, centroid in enumerate(centroids):
            temp_distance = point.manhatten_distance(centroid)
            if point.cluster == -1 or point.distance > temp_distance:
                point.cluster = i
                point.distance = temp_distance
    new_clusters = [x.cluster for x in points]
    for a, b in zip(old_clusters, new_clusters):
        if a != b:
            changed += 1
    return changed

def calculate_average_error(centroids: list[Point], points: list[Point]):
    error = 0.0
    for point in points:
        final_dis = False
        for i, centroid in enumerate(centroids):
            temp_dis = point.manhatten_distance(centroid)
            if not final_dis:
                final_dis = temp_dis
            if temp_dis < final_dis:
                final_dis = temp_dis
        error += final_dis
    return error
            

def get_clustered_data(points: list[Point]):
    clustered_data = {}
    for i, point in enumerate(points):
        try:
            clustered_data[point.cluster].append(i)
        except KeyError:
            clustered_data[point.cluster] = [i]
    return clustered_data

def change_medoid(centroids: list[Point], points: list[Point]):
    prev_error = calculate_average_error(centroids, points)
    clustered_data = get_clustered_data(points)
    for i, centroid in enumerate(centroids):
        new_centroids = deepcopy(centroids)
        for j in clustered_data[i]:
            new_centroids[i] = points[j]
            new_error = calculate_average_error(new_centroids, points)
            if new_error < prev_error:
                return assign_cluster(new_centroids, points), new_centroids
    return 0, centroids


def run_recursive_step(centroids: list[Point], points: list[Point]):
    changed = -1
    while changed != 0:
        changed, centroids = change_medoid(centroids, points)


def run_k_medoids(data: list, k: int):
    centroids = []
    points = []
    for centroid in data[:k]:
        centroids.append(Point(deepcopy(centroid[:-1])))

    for point in data:
        points.append(Point(point[:-1]))

    assign_cluster(centroids, points)
    run_recursive_step(centroids, points)
    return [x.attributes for x in points], [x.cluster for x in points]


if __name__ == "__main__":
    file = "iris.data"
    data = read_file(file)
    data = process_data(data)
    data = prepare_data(data)
    random.shuffle(data)
    points, clusters = run_k_medoids(data, 3)
    print(silhouette_score(points, clusters))
