import dlib
import numpy as np
from skimage import io



imagepath = "/home/fzwu/Desktop/082498.jpg"
image = io.imread(imagepath)
face_detector = dlib.get_frontal_face_detector()
detected_face = face_detector(image, 1)
d = detected_face[0]
bbox = np.array([d.left(), d.top(), d.right(), d.bottom()])
fid = open("test.npy", "wb")
np.save(fid, bbox)








