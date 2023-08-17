import random
import os
import numpy
from shutil import rmtree
import torch
from common import read_file


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def remove_key(d, key):
    r = dict(d)
    del r[key]
    return r


def get_diseases(base_dir):
	unique_tags_diseases = {}
	with open(os.path.join(base_dir, "ImageCLEFmedical_Caption_2023_cui_mapping.csv"), mode='r') as file:
		cui_mappings = file.read().split("\n")
	for entry in cui_mappings:
		cui_mapping = entry.split("\t")
		unique_tags_diseases[cui_mapping[0]] = cui_mapping[1]
	return unique_tags_diseases


def split_cases(images, keys, filename):
	new_images = {}
	if keys is not None:
		for key in keys:
			new_images[key] = images[key]

		try:
			with open(filename, "w") as output_file:
				for new_image in new_images:
					output_file.write(new_image + "\t" + new_images[new_image])
					output_file.write("\n")
			return new_images
		except:
			return new_images


def decompose(file_entries):
	data_container = {}
	for entry in file_entries:
		try:
			entry_decomposed = entry.split(",")
			cuis_decomposed = entry_decomposed[1].split(";")
			data_container[entry_decomposed[0] + ".jpg"] = cuis_decomposed
		except:
			pass
	return data_container


def get_pkaliosis_splits(task='tagging'):

	if task == 'tagging':
		pkaliosis_splits_base = "/media/georg_mosh/Samsung_T51/ImageCLEFmedical/pkaliosis_splits"
		train_data = decompose(read_file(pkaliosis_splits_base, "imageclef2023_concepts_train_split.csv"))
		val_data = decompose(read_file(pkaliosis_splits_base, "imageclef2023_concepts_valid_split.csv"))
		dev_data = decompose(read_file(pkaliosis_splits_base, "imageclef2023_concepts_test_split.csv"))
	else:
		raise Exception("Please use get_captioning_splits method instead.")

	return train_data, val_data, dev_data


def get_captioning_splits(captions, train_set, val_set, dev_set):
	# Create the train split.
	train_data = {}
	for image_id in train_set.keys():
		train_data[image_id] = captions[image_id[:-4]]

	# Create the validation split.
	val_data = {}
	for image_id in val_set.keys():
		val_data[image_id] = captions[image_id[:-4]]

	# Create the development split.
	dev_data = {}
	for image_id in dev_set.keys():
		dev_data[image_id] = captions[image_id[:-4]]

	return train_data, val_data, dev_data

def get_imageclef(val_perc=0.1, dev_perc=0.1, task='captioning', base='', year=2022):
	set_seeds(0)

	if task == 'captioning':
		# create dataset folder
		try:
			rmtree(base + "imageclef2022/captioning/")
		except BaseException:
			pass
		# os.makedirs(base + "imageclef2022/captioning/")

		# read the reports xml files and create the dataset tsv
		# read the reports xml files and create the dataset tsv
		if year == 2022:
			reports_train_path = base + "imageclef2022/c856ae07-029b-449e-bd06-99c04d3ad1e0_ImageCLEFmedCaption_2022_caption_prediction_train.csv"
			reports_val_path = base + "imageclef2022/cc3d9c72-6c2b-4bd3-9d10-4e133031be48_ImageCLEFmedCaption_2022_caption_prediction_valid.csv"
		elif year == 2023:
			reports_train_path = os.path.join(base, "ImageCLEFmedical_Caption_2023_caption_prediction_train_labels.csv")
			reports_val_path = os.path.join(base, "ImageCLEFmedical_Caption_2023_caption_prediction_valid_labels.csv")
		else:
			print("Invalid Year")
			raise Exception("Datasets available are for 2022-2023.")


		# load data
		images = {}

		with open(reports_train_path, "r") as file:
			for line in file:
				line = line.replace("\n", "").split("\t")
				images[line[0] + ".jpg"] = line[1]

		with open(reports_val_path, "r") as file:
			for line in file:
				line = line.replace("\n", "").split("\t")
				images[line[0] + ".jpg"] = line[1]

		# split data
		set_seeds(0)
		images = remove_key(images, "ID.jpg")
		keys = list(images.keys())  # sort keys?
		random.shuffle(keys)

		train_perc = 1 - (val_perc + dev_perc)
		train_split = int(numpy.floor(len(images) * train_perc))
		val_split = int(numpy.floor(len(images) * val_perc))
		train_keys = keys[:train_split]  # training set
		val_keys = keys[train_split:train_split+val_split]  # validation set
		dev_keys = keys[train_split+val_split:]  # development set

		train_set = split_cases(images, train_keys, base + "imageclef2023/captioning/train_images.tsv")
		val_set = split_cases(images, val_keys, base + "imageclef2023/captioning/val_images.tsv")
		dev_set = split_cases(images, dev_keys, base + "imageclef2023/captioning/dev_images.tsv")

		print("train size", len(train_keys) / len(keys))
		print("validation size", len(val_keys) / len(keys))
		print("development size", len(dev_keys) / len(keys))

		return train_set, val_set, dev_set

	elif task == 'classification':
		# create dataset folder
		try:
			# rmtree(base + "imageclef2022/classification/")
			rmtree("./classification/")
		except BaseException:
			pass
		# os.makedirs(base + "imageclef2022/classification/")
		os.makedirs("./classification/")

		# read the reports xml files and create the dataset tsv
		if year == 2022:
			reports_train_path = base + "imageclef2022/b47c4f80-9432-408c-b69a-956a3382a0da_ImageCLEFmedCaption_2022_concept_detection_train.csv"
			reports_val_path = base + "imageclef2022/46bff9d5-95d4-4362-be98-ef59819ec3af_ImageCLEFmedCaption_2022_concept_detection_valid.csv"
		elif year == 2023:
			reports_train_path = os.path.join(base, "ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv")
			reports_val_path = os.path.join(base, "ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv")
		else:
			print("Invalid Year")
			raise Exception("Datasets available are for 2022-2023.")

		# load data
		images = {}

		with open(reports_train_path, "r") as file:
			for line in file:
				line = line.replace("\n", "").split("\t")
				tags = line[1].split(";")
				images[line[0] + ".jpg"] = tags

		with open(reports_val_path, "r") as file:
			for line in file:
				line = line.replace("\n", "").split("\t")
				tags = line[1].split(";")
				images[line[0] + ".jpg"] = tags

		# split data
		set_seeds(0)
		images = remove_key(images, "ID.jpg")
		keys = list(images.keys())  # sort keys?
		random.shuffle(keys)

		train_perc = 1 - (val_perc + dev_perc)
		train_split = int(numpy.floor(len(images) * train_perc))
		val_split = int(numpy.floor(len(images) * val_perc))
		train_keys = keys[:train_split]  # training set
		val_keys = keys[train_split:train_split + val_split]  # validation set
		dev_keys = keys[train_split + val_split:]  # development set

		if year == 2022:
			train_set = split_cases(images, train_keys, base + "imageclef2022/classification/train_images.tsv")
			val_set = split_cases(images, val_keys, base + "imageclef2022/classification/val_images.tsv")
			dev_set = split_cases(images, dev_keys, base + "imageclef2022/classification/dev_images.tsv")
		else:
			train_set = split_cases(images, train_keys, "./classification/train_images.tsv")
			val_set = split_cases(images, val_keys, "./classification/val_images.tsv")
			dev_set = split_cases(images, dev_keys, "./classification/dev_images.tsv")

		print("train size", len(train_keys) / len(keys))
		print("validation size", len(val_keys) / len(keys))
		print("development size", len(dev_keys) / len(keys))

		row_id = 0
		unique_tags = {}
		for key in keys:
			image_tags = images[key]
			for tag in image_tags:
				if tag not in unique_tags.keys():
					unique_tags[tag] = row_id
					row_id += 1

		return train_set, val_set, dev_set, unique_tags


def get_imageclef_clsmix(val_perc=0.1, dev_perc=0.1, task='captioning', base='', file=''):

		# create dataset folder
		try:
			rmtree(base + "classification/")
		except BaseException:
			pass
		os.makedirs(base + "classification/")

		# read the reports xml files and create the dataset tsv
		reports_train_path = os.path.join(base, file)

		# load data
		images = {}

		with open(reports_train_path, "r") as file:
			for line in file:
				try:
					line = line.replace("\n", "").split("\t")
					tags = line[1].split(";")
					images[line[0]] = tags
				except:
					continue

		# split data
		set_seeds(0)
		keys = list(images.keys())  # sort keys?

		train_perc = 1 - (val_perc + dev_perc)
		train_split = int(numpy.floor(len(images) * train_perc))
		val_split = int(numpy.floor(len(images) * val_perc))

		train_keys = keys[:train_split]  # training set
		val_keys = keys[train_split:train_split + val_split]  # validation set
		dev_keys = keys[train_split + val_split:]  # development set


		train_set = split_cases(images, train_keys, os.path.join(base, "classification/train_images.tsv"))
		val_set = split_cases(images, val_keys, os.path.join(base, "classification/val_images.tsv"))
		dev_set = split_cases(images, dev_keys, os.path.join(base, "classification/dev_images.tsv"))

		print("train size", len(train_keys) / len(keys))
		print("validation size", len(val_keys) / len(keys))
		print("development size", len(dev_keys) / len(keys))

		row_id = 0
		unique_tags = {}
		for key in keys:
			image_tags = images[key]
			for tag in image_tags:
				if tag not in unique_tags.keys():
					unique_tags[tag] = row_id
					row_id += 1

		return train_set, val_set, dev_set, unique_tags

def get_test_imageclef(task=None, base=''):

	# load data
	test_set = os.listdir(base + "imageclef2022/imageclef2022_images_test")
	test_set = list(test_set)

	return test_set

