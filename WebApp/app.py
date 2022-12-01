# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 18:08:08 202f

@author: Sara
"""
import os
from flask import Flask, render_template, request
from keras.models import load_model 
import skimage 
import cv2 
import numpy as np 
from skimage.transform import resize  
import scipy     
from sklearn.cluster import KMeans  
from skimage.filters import threshold_otsu    
from PIL import Image 
import random  
#import matplotlib.pyplot as plt   
#%%
clusters = 2  
kernel_size = 5
iterations = 2
path = os.getcwd()+'/static/'
filelist = [ f for f in os.listdir(path) if f.endswith(".png") ]
for f in filelist:
     os.remove(os.path.join(path, f))   
app = Flask(__name__) 
# Tune image for CNN model
def tune_image(image):
    img_resized = resize(image, (28,28), anti_aliasing=True, mode='reflect').reshape((-1, 1, 28, 28)).reshape(28,28) 
    return np.asarray([cv2.cvtColor(np.float32(img_resized), cv2.COLOR_BGRA2RGB)])
# Tune image for Unet model
def tune_image_unet(image):
    image = cv2.resize(cv2.cvtColor(image.reshape((256,256))  , cv2.COLOR_GRAY2RGB), (299, 299))
    padding_y = (299-image.shape[0])/2
    return np.pad(image[:,:,2] , ((int(padding_y),int(padding_y)),(0,0)), mode='constant', constant_values=0)
# Check image file name(Covid or normal)
def check_image(filename):
    patch = ""
    if "COVID" in filename:
        name = filename.split('COVID')[-1]
        patch = name.rsplit('.png')[0] 
    if "Normal" in filename:
        name = filename.split('Normal')[-1]
        patch = name.rsplit('.png')[0] 
    return patch, name
# Return prediction as string
def check_covid(label):
   if label == 0:
    return "The person has covid"
   if label == 1:
    return "The person is normal"
# Save image into path, to load on website
def save_image(image, model_name="",patch="",fil=""):   
     if fil == "": 
      image.save(path+""+ model_name+""+patch+"")
      return "static/"+model_name+""+patch+""
     else:
      image.save(path+""+fil+"_"+ model_name+""+patch+"")  
      return "static/"+fil+"_"+model_name+""+patch+""    
# Run salt and pepper noise
def salt_pepper(image,prob): 
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
# Choose filter, then apply on image
def filtr(image, fil =""):
    if fil == "":
        return image
    elif fil == "Bilateral filter":
        return cv2.bilateralFilter(image, 15, 75, 75)
    elif fil =="Salt and pepper noise":
        return salt_pepper (image,0.05)
    elif fil == "Median filter":
        return cv2.medianBlur(image, 3)   
# Thresholding        
def binary(image):
    return image > threshold_otsu(image)
# Mean filter  
def mean_filter(arr, k):
    p = len(arr)
    diag_offset = np.linspace(-(k//2), k//2, k, dtype=int)
    eL = scipy.sparse.diags(np.ones((k, p)), offsets=diag_offset, shape=(p, p))
    nrmlize = eL @ np.ones_like(arr)
    return (eL @ arr) / nrmlize
# Merge mask with image           
def mask_merge_segmented(uri_image, image_cluster):
    new_image = uri_image.copy() 
    new_image[:,:]  = new_image[:,:]* image_cluster
    return new_image
# Select cluster index for image 
# Check cluster having smallest mean
def select_cluster_index(clusters):
     minx = clusters[0].mean()
     index = 0
     for i in clusters:
         if i.mean() < minx:
             minx = i.mean()
             index += 1
     return index 
# Test image for u-net segmentation
def test_on_image(model, img_array, img_num, img_side_size = 256):
    pred = model.predict(img_array.reshape(1,img_side_size,img_side_size,1))
    pred[pred>0.5] = 1.0
    pred[pred<0.5] = 0.0   
    return pred 
# Predict original image using CNN
def original(path, patch, fil = ""): 
    model = load_model('models/cnn.h5')
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics = ["accuracy"])
    original = skimage.io.imread(path) 
    original = filtr(original,fil)
    image = tune_image(original)
    y_pred = model.predict(image, batch_size=32)
    label =  int(y_pred>=0.5) 
    p = check_covid(label)  
    image_tosave = Image.fromarray(original) 
    path = save_image(image_tosave,'original',patch,fil)         
    return p, path   
# Segment image using k-means then predict (comments in details are found within experiments project)
def k_means_cl(path, patch, fil = ""):
    model = load_model('models/cnn_segmented_kmeans.h5')
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics = ["accuracy"])
    original = skimage.io.imread(path)  
    k_means_image = KMeans(random_state=1, 
                  n_clusters = 2, 
                  init='k-means++'
    ).fit(original.reshape((-1,1))).labels_.reshape(original.shape)    
    clusters_colors_image = [k_means_image == i for i in range(clusters)]
    image_mean_filter = mean_filter(clusters_colors_image[select_cluster_index(clusters_colors_image)]    , 20)
    mask_binary = binary(image_mean_filter)     
    image_result = mask_merge_segmented(original ,mask_binary) 
    image_result = filtr(image_result,fil)
    image = tune_image(image_result) 
    y_pred = model.predict(image,batch_size=32)
    label =  int(y_pred>=0.5)    
   # plot_images([original, image_result]) 
    image_tosave = Image.fromarray(image_result) 
    k_means_path = save_image(image_tosave,'kmeans',patch,fil)  
    return check_covid(label), k_means_path 
# Segment image using flood then predict (comments in details are found within experiments project)
def flood(path, patch, fil = ""):
    model = load_model('models/cnn_segmented_flood.h5')
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics = ["accuracy"])
    original = skimage.io.imread(path) 
    original_ch = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)  
    image_float = original_ch.astype(float) / 255.
    k_channel = 1 - np.max(image_float, axis=2)
    k_channel = (255*k_channel).astype(np.uint8) 
    _, mask_binary = cv2.threshold(k_channel, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask_binary = binary(cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, morph_kernel, None, None, iterations, cv2.BORDER_REFLECT101))
    image_result = mask_merge_segmented(original ,mask_binary) 
    image_result = filtr(image_result,fil) 
    image = tune_image(image_result)
    y_pred = model.predict(image,batch_size=32)
    label =  int(y_pred>=0.5)   
    #plot_images([original, image_result]) 
    image_tosave = Image.fromarray(image_result) 
    path = save_image(image_tosave,'flood',patch,fil)         
    return check_covid(label) , path  
# Segment image using u-net then predict (comments in details are found within experiments project)
def unet(path, patch, fil = ""):
    unet_model =  load_model('models/unet.h5', compile = False)
    unet_model.compile(loss="binary_crossentropy", optimizer="adam",metrics = ["accuracy"])
    original = skimage.io.imread(path) 
    resized_image = cv2.resize(original,(256,256)) 
    mask_pred = test_on_image(unet_model, img_array = resized_image, img_num = 3, img_side_size = 256)
    mask_binary = tune_image_unet(mask_pred)
    image_result = mask_merge_segmented(original, mask_binary)    
    #plot_images([original, padded_img, image_result])
    model = load_model('models/cnn_segmented_unet.h5')
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics = ["accuracy"]) 
    image_res = filtr(image_result,fil)
    image = tune_image(image_res)
    y_pred = model.predict(image,batch_size=32)
    label =  int(y_pred>=0.5)   
    #plot_images([original, image_result]) 
    image_tosave = Image.fromarray(image_res) 
    path = save_image(image_tosave,'unet',patch,fil)         
    return check_covid(label), path  
#%% Routes whenever form submitted.
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html") 
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        # Get image input
        img = request.files['my_image']
        # Get dropdown input
        select = request.form.get('my_option')
        # Get image file name #, to retrieve the result with same #
        name, patch = check_image(img.filename)
        #Save image
        save_image(img, img.filename, "","")
        # Get filter/noise option
        option = str(select)
        img_path =  path+"" + img.filename    
        # Get predictions of 4 models
        p1, i1 = original(img_path,patch)  
        p2, i2 = k_means_cl(img_path,patch) 
        p3, i3 = unet(img_path,patch) 
        p4, i4 = flood(img_path,patch)  
        # If filter is specified, get other predictions for filtered images too
        if option == "Filter Or Noise":
          return render_template("index.html", prediction1 = p1, prediction2 = p2, prediction3 = p3, prediction4 = p4, img_path1 = i1, img_path2 = i2 , img_path3 = i3, img_path4 = i4 )
        else: 
            p1f, i1f = original(img_path,patch, option)   
            p2f, i2f = k_means_cl(img_path,patch, option) 
            p3f, i3f = unet(img_path,patch, option) 
            p4f, i4f = flood(img_path,patch, option) 
            return render_template("index.html", prediction1 = p1, prediction2 = p2, prediction3 = p3, prediction4 = p4, img_path1 = i1, img_path2 = i2 , img_path3 = i3, img_path4 = i4 ,  prediction1f = p1f, prediction2f = p2f, prediction3f = p3f, prediction4f = p4f, img_path1f = i1f, img_path2f = i2f , img_path3f = i3f, img_path4f = i4f , options = option)
#%% Run app
if __name__ =='__main__':
	#app.debug = True
	app.run()