How to run experiment:
1) Go to experiments folder
2) There are 3 files for segmentation (flood_segmentation.py, unet_segmentation.py, kmeans_segmentation.py )
3) Choose the file you need for segmentation type to obtain segmented images
4) Segmented images are now ready to be classified
5) We test several DL and ML models to classify images (classification.py)

What if I need to run experiment on single image using best model? 
Run WebApp (classifies images using CNN):
1) On spyder, run app.py
2) Go to 127.0.0.1:5000/
3) Choose an x-ray image
4) To check the behaviour of model against filters, choose a certain filter
5) Then submit to display result
