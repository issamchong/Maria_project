
# import the needed libraries
from tensorflow.keras.models import load_model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
# read the input image using Pillow (you can use another library, e.g., OpenCV)
img1= Image.open("/home/issam/ML/Maria_project/dataset/testing/desk/desk_0217.png")
img2= Image.open("/home/issam/ML//Maria_project/dataset/testing/desk/desk_0219.png")
# convert to ndarray numpy
img1 = np.asarray(img1)
img2 = np.asarray(img2)


# load the trained model
model = load_model('/home/issam/ML/Maria_project/Maria_model.h5')

# predict the input image using the loaded model
pred1 = model.predict_classes((img1/255).reshape((1,357,90,1)))
pred2 = model.predict_classes((img2/255).reshape((1,357,90,1)))

if ( pred1[0]==1):
    print("it is a desk")

# plot the prediction result
plt.figure('img1')
plt.imshow(img1,cmap='gray')
plt.title('pred:'+str(pred1[0]), fontsize=22)

plt.figure('img2')
plt.imshow(img2,cmap='gray')
plt.title('pred:'+str(pred2[0]), fontsize=22)

plt.show()
