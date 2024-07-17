import numpy as np
from scipy.spatial import distance

# def manhattan_distance(v1, v2):
#     """_summary_
#     Compute manhattan (cityblock) distance
#     dist = sum(|x_i - y_i|)
#     dist = distance.cityblock(v1, v2)
#     Args:
#         v1 (_type_): query image signature
#         v2 (_type_): offline signatures
#     """
#     return np.sum(np.abs(np.array(v1) - np.array(v2)))

def manhattan_distance(v1, v2):
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    return np.sum(np.abs(v1 - v2))


def euclidean_distance(v1, v2):
    """_summary_
    Compute manhattan distance
    dist = sqrt(sum(|x_i - y_i|)**2)
    dist = distance.euclidean(v1, v2)
    Args:
        v1 (_type_): query image signature
        v2 (_type_): offline signatures
    """
    return np.sqrt(np.sum((np.array(v1) - np.array(v2).astype('float'))**2))

def chebyshev_distance(v1, v2):
    """_summary_
    Compute manhattan distance
    dist = max(|x_i - y_i|)
    dist = distance.chebyshev(v1, v2)
    Args:
        v1 (_type_): query image signature
        v2 (_type_): offline signatures
    """
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    return np.max(np.abs(v1 - v2))

def canberra_distance(v1, v2):
    """_summary_
    Compute manhattan distance
    
    Args:
        v1 (_type_): query image signature
        v2 (_type_): offline signatures
    """
    return distance.canberra(v1, v2)

def retrieve_similar_images(features_db, query_features, distance, num_results):
    distances = []
    
    for instance in features_db:
        features, label, img_path = instance[:-2],instance[-2], instance[-1]
        print(f'{features}- {label} - {img_path}')
        if distance == 'euclidean':
            dist = euclidean_distance(query_features, features)
        if distance == 'manhattan':
            dist = manhattan_distance(query_features, features)
        if distance == 'chebyshev':
            dist = chebyshev_distance(query_features, features)
        if distance == 'canberra':
            dist = canberra_distance(query_features, features)
        distances.append((img_path, dist, label))
    distances.sort(key=lambda x: x[1])
    return distances[:num_results]