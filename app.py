from flask import Flask, render_template, request
# from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
# from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv

app = Flask(__name__)

dic = {0 : 'Blur', 1 : 'Clear'}

model = load_model('model.h5')

model.make_predict_function()

# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,3)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]
def load_image(img_path, show=True):

    # img = image.load_img(img_path, target_size=(200, 150))
    # img_tensor = image.img_to_array(img)                    # (height, width, channels)
    # img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    # img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    # print(img_tensor.shape)

    # =================================================================================================================

    img = cv.imread(img_path,0)

    edges = cv.Canny(img,100,200)
    edges = ~edges
    edges = cv.cvtColor(edges, cv.COLOR_BGR2RGB)
    edges = cv.resize(edges, (150,200))

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]),plt.yticks([])
    plt.show()
    img_tensor = np.expand_dims(edges, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    print(img_tensor.shape)
    img_tensor = img_tensor.astype('float64')
    img_tensor /= 255.
	# p = model.predict(img_tensor)
	# return dic[p[0]]


    # if show:
    #     plt.imshow(img_tensor[0])                           
    #     plt.axis('off')
    #     plt.show()

    return img_tensor

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = load_image(img_path)
		pred = model.predict(p)
		print(pred)
		if(pred>0.5):
			p="clear"
		else:
			p="blur"


	return render_template("index.html", prediction = p, img_path = img_path,pre=pred)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)