import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import shutil
from PIL import Image
from distances import retrieve_similar_images
from data_processing import extract_features
import os

# Load offline signatures
# signatures_glcm = np.load('./Signatures/signatures_glcm.npy')
# signatures_bit = np.load('./Signatures/signatures_bitdesc.npy')
# signatures_haralick = np.load('./Signatures/signatures_haralick.npy')
# signatures_haralick_glcm = np.load('./Signatures/signatures_haralick_glcm.npy')
# signatures_haralick_bit = np.load('./Signatures/signatures_haralick_BiT.npy')

signatures_glcm = np.load('./Sign_glcm.npy')
signatures_bit = np.load('./Sign_bit.npy')
signatures_haralick = np.load('./Sign_haralick.npy')
signatures_haralick_glcm = np.load('./Sign_haralick_glcm.npy')
signatures_haralick_bit = np.load('./Sign_haralick_bit.npy')

def display_images_side_by_side(image_paths, captions, width=200):
    num_images = len(image_paths)
    num_columns = st.columns(num_images)
    for i in range(num_images):
        with num_columns[i]:
            st.image(image_paths[i], caption=captions[i], width=width)

def main():
    st.title("Résultats de la recherche d'images similaires")

    uploaded_file = st.sidebar.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        # Enregistrer l'image téléchargée sur le disque
        img_path = "uploaded_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        img = Image.open(uploaded_file)
        st.image(img, caption="Image de requête", use_column_width=True)

        # Sélection des signatures
        selected_signatures = st.sidebar.selectbox("Sélectionnez les signatures à utiliser",
                                                   ["GLCM", "Bitdesc", "Haralick", "Haralick_GLCM", "Haralick_Bitdesc"])

        selected_distance = st.sidebar.selectbox("Sélectionnez les distances à utiliser",
                                                  ["Euclidean", "Manhattan", "Chebyshev", "Canberra"])

        if selected_signatures == "GLCM":
            selected_features = signatures_glcm
            selected_extract = "GLCM"
        elif selected_signatures == "Bitdesc":
            selected_features = signatures_bit
            selected_extract = "Bitdesc"
        elif selected_signatures == "Haralick":
            selected_features = signatures_haralick
            selected_extract = "Haralick"
        elif selected_signatures == "Haralick_GLCM":
            selected_features = signatures_haralick_glcm
            selected_extract = "Haralick_GLCM"
        elif selected_signatures == "Haralick_Bitdesc":
            selected_features = signatures_haralick_bit
            selected_extract = "Haralick_Bitdesc"

        if selected_distance == "Euclidean":
            selected_distances = "euclidean"
        elif selected_distance == "Manhattan":
            selected_distances = "manhattan"
        elif selected_distance == "Chebyshev":
            selected_distances = "chebyshev"
        elif selected_distance == "Canberra":
            selected_distances = "canberra"

        features = extract_features(img_path, selected_extract)

        result = retrieve_similar_images(features_db=selected_features, query_features=features,
                                         distance=selected_distances, num_results=10)
        

        st.subheader("Résultats :")
        similarity_folder = "Similarity"
        if not os.path.exists(similarity_folder):
            os.makedirs(similarity_folder)
        temp_similarity_paths = []
        image_paths = []
        captions = []
        distances = []  # Stocker les distances pour le diagramme
        for img_name, dist, label in result:
            img_path = f"datasets/{img_name}"
            if os.path.exists(img_path):
                image_paths.append(img_path)
                captions.append(f"Distance : {dist:.2f}, Étiquette : {label}, Path : {img_path}")
                distances.append(dist)  # Ajouter la distance à la liste
            else:
                st.error(f"Erreur : Le chemin de l'image {img_name} est introuvable.")
        

        # Afficher les images côte à côte avec une largeur de 200 pixels chacune
        display_images_side_by_side(image_paths, captions, width=60)

        # Créer un diagramme des distances
        st.subheader("Diagramme des distances :")
        fig, ax = plt.subplots()
        ax.bar(range(len(distances)), distances, color='skyblue')
        ax.set_xlabel('Images')
        ax.set_ylabel('Distances')
        ax.set_title('Distances entre l\'image de requête et les autres images')
        st.pyplot(fig)

        # Supprimer l'image téléchargée après l'utilisation
        os.remove(img_path)

if __name__ == '__main__':
    main()
