from skimage.feature import graycomatrix, graycoprops
from BiT import bio_taxo
from mahotas.features import haralick
import cv2


def glcm(data):
    glcm = graycomatrix(data, [2], [0], None, symmetric=True, normed=True)
    diss = graycoprops(glcm, 'dissimilarity')[0,0]
    cont = graycoprops(glcm, 'contrast')[0,0]
    corr = graycoprops(glcm, 'correlation')[0,0]
    ener = graycoprops(glcm, 'energy')[0,0]
    homo = graycoprops(glcm, 'homogeneity')[0,0]
    return [diss, cont, corr, ener, homo]


def bitdesc(data):
    return bio_taxo(data)


# def haralick_features(image):
#     print("Image path:", image)
#     data = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
#     haralick_features = haralick(data.astype(int))
#     return haralick_features.mean(axis=0).tolist()

def haralick_features(image):
    """Extract Haralick features from the given image."""
    haralick_features = haralick(image.astype(int))
    return haralick_features.mean(axis=0).tolist()


def haralick_glcm(image):
    return haralick_features(image)+glcm(image)


def haralick_BiT(image):
    return haralick_features(image)+bitdesc(image)