import cv2, numpy as np, face_recognition, os


# ImageDB path
path = './datasets'


# Global variables
image_list = []
name_list = []
#Gral all images from the folder
mylist = os.listdir(path)
# print(mylist)

# Load images
for img in mylist:
    if os.path.splitext(img)[1].lower() in ['.jpg', '.png', '.jpeg']:
        curImg = cv2.imread(os.path.join(path,img))
        image_list.append(curImg)
        imgName = os.path.splitext(img)[0]
        name_list.append(imgName)


# Define a function to detecct face and extract features therefrom
def findEncodings(img_list, ImgName_list):
    """_summary_
    Define a function to detect face and extract features therefrom

    Args:
        imh_list (list): List of BGR of images
        ImgName_list (list): List of image names
    """

    signatures_db = []
    count = 1
    for myImg, name in zip(img_list, ImgName_list):
        img = cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB)
        signature = face_recognition.face_encodings(img)[0]
        signature_class = signature.tolist() + [name]
        signatures_db.append(signature_class)
        print(f'{int((count / len(img_list))) * 100} % extracted')
        count+=1
    signatures_db = np.array(signatures_db)
    np.save('Signatures.npy', signatures_db)
    print('Signature_db stored')


def main():
    findEncodings(image_list, name_list)


if __name__ == '__main__':
    main()
