from PIL import Image
from skimage import data, transform
import os


# Image Prep Location

ROOT_PATH = "/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/"

PREP_train_data_directory = os.path.join(ROOT_PATH, "PrepImageArea/Training")
PREP_test_data_directory = os.path.join(ROOT_PATH, "PrepImageArea/Testing")

train_data_directory = os.path.join(ROOT_PATH, "TidePods/Training")
test_data_directory = os.path.join(ROOT_PATH, "TidePods/Testing")


# Getting Images from Prep Location
def prep_images(PREP_DIR, REAL_DIR):
	images = []
	image_filenames = []
	image_labels = []
	image_paths = []

	for d in os.listdir(PREP_DIR):
		label_directory_path = os.path.join(PREP_DIR, d)
		temp_label = d;
		
		if os.path.isdir(label_directory_path):
			temp_image_paths = []
			
			# getting file correct file paths 
			for f in os.listdir(label_directory_path):
				if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".gif"):
					temp_image_paths.append(os.path.join(label_directory_path, f))
					image_paths.append(os.path.join(label_directory_path, f))
					print(os.path.join(label_directory_path, f))
					image_filenames.append(os.path.splitext(f)[0])
					image_labels.append(temp_label)
			 
			# reading in image data from paths
			for f in temp_image_paths:
				images.append(Image.open(f))
				
				
	# convert images to RGB and save them as JPG

	counter = 0
			
	for i in images:
		i.convert('RGB').save(REAL_DIR+"/"+image_labels[counter]+"/"+image_filenames[counter]+"-MODIFIED.jpg")
		counter += 1

	# delete image files from prep location
	for i in image_paths:
		os.remove(i)


prep_images(PREP_train_data_directory, train_data_directory)


