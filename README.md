How to run experiment:
1) Go to experiments folder
2) 3 files for segmentation (flood_segmentation.py, unet_segmentation.py, kmeans_segmentation.py )
3) Choose the file you need for segmentation type to obtain segmented images
4) Segmented images are now ready to be classified
5) Test several DL and ML models to classify images and obtain results (classification.py)
 
Run WebApp (Using flask) to run experiment on single image using best model (classifies images using CNN):
1) On spyder, run app.py
2) Go to 127.0.0.1:5000/
3) Choose an x-ray image
4) To check the behaviour of model against filters, choose a certain filter
5) Submit x-ray and filter to display result

For training U-net segmentation model on masks:
1) Download the CXR_png, masks and test folders from https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset/data
2) Copy the folders (CXR_png, masks, test) under \Experiments\data\unsegmented + unet masks\ 

For Data analysis and evaluation: 
1) Download the Normal and Covid data without or without masks from https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2) Copy the folders (COVID, Normal) under \Experiments\data\unsegmented\
