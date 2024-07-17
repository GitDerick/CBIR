import os
import cv2
import numpy as np
from descriptors import glcm, bitdesc, haralick_features, haralick_glcm, haralick_BiT
from tqdm import tqdm





def extract_features(image_path, descriptor):
#def extract_features(image_path):
    """Extracts features from a grayscale image using the specified descriptor."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if img is not None:
    #     features = haralick_BiT(img)
    #     return features
    # else:
    #     print(f"Failed to load image: {image_path}")
    #     return [None*5]
    if img is not None:
        if descriptor == "GLCM":
            features = glcm(img)
        elif descriptor == "Bitdesc":
            features = bitdesc(img)
        elif descriptor == "Haralick":
            features = haralick_features(img)
        elif descriptor == "Haralick_GLCM":
            features = haralick_glcm(img)
        elif descriptor == "Haralick_Bitdesc":
            features = haralick_BiT(img)
        else:
            raise ValueError("Descriptor not supported")
        
        return features
    else:
        return None


def process_datasets(root_folder):
    """Process all images in the dataset folder and create signatures.

    Args:
        root_folder (str): Chemin du dossier racine contenant les images.
    """
    all_features = []  # Liste pour stocker toutes les caractéristiques et métadonnées
    total_files = sum([len(files) for _, _, files in os.walk(root_folder) if files])  # Nombre total de fichiers à traiter
    count = 1  # Compteur pour suivre le nombre de fichiers traités
    with tqdm(total=total_files, desc="Processing images") as pbar:  # Initialiser la barre de progression
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    # Construire le chemin relatif et extraire les caractéristiques
                    relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                    features = extract_features(os.path.join(root, file))
                    if features is not None:
                        # Extraire le nom de classe du chemin relatif
                        class_name = os.path.basename(os.path.dirname(relative_path))
                        # Vérifier si features est déjà une liste
                        if isinstance(features, list):
                            all_features.append(features + [class_name, relative_path])
                        else:
                            all_features.append(features.tolist() + [class_name, relative_path])
                    pbar.update(1)  # Mettre à jour la barre de progression
                    # print(f'{int((count / total_files) * 100)} % extracted')  # Afficher le pourcentage extrait
                    count += 1  # Incrémenter le compteur
    signatures = np.array(all_features)
    np.save('Sign_haralick_bit.npy', signatures)
    print('Features successfully stored.')



def main():
    process_datasets('./datasets')

if __name__ == '__main__':
    main()

