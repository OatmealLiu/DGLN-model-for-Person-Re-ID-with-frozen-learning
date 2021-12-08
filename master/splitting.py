import os
import pandas as pd
import random

def create_random_split_train_val_attributes(root_dir):
    def extract_person_id(image_name):
        person_id = image_name.split('_')[0]
        return int(person_id)

    csv_file = os.path.join(root_dir, "annotations_train.csv")
    annotations = pd.read_csv(csv_file, index_col="id")

    validation_frac = 0.2
    validation_split = annotations.sample(frac=validation_frac)

    train_dir = os.path.join(root_dir, "train")
    validation_dir = os.path.join(root_dir, "val")
    validation_images_names = [image_name for image_name in os.listdir(train_dir) if
                               extract_person_id(image_name) in validation_split.index]

    os.makedirs(validation_dir, exist_ok=True)
    for image_name in validation_images_names:
        old_path = os.path.join(train_dir, image_name)
        new_path = os.path.join(validation_dir, image_name)
        os.rename(old_path, new_path)

def create_random_split_train_val_ID(root_dir):
    def extract_person_id(image_name):
        person_id = image_name.split('_')[0]
        return int(person_id)

    trainID_dir = os.path.join(root_dir, "train_id")

    valID_dir = os.path.join(root_dir, "val_id")
    os.makedirs(valID_dir, exist_ok=True)

    # split unique image of each distinct person in gallery_train dir, move it to queries_train dir
    images_names = os.listdir(trainID_dir)
    grouped_images = {}
    for img_name in images_names:
        person_id = extract_person_id(img_name)
        if person_id not in grouped_images:
            grouped_images[person_id] = [img_name]
        else:
            grouped_images[person_id].append(img_name)

    val_img_names = []
    for _, unique_id_imgs in grouped_images.items():
        unique_img_size = len(unique_id_imgs)
        size2val = int(0.2 * unique_img_size)

        selected = set()
        stop = 0
        while stop < size2val:
            random_img = random.randint(0,-1+unique_img_size)
            if random_img in selected:
                continue
            else:
                val_img_names.append(unique_id_imgs[random_img])
                selected.add(random_img)
                stop += 1

    # for each distinct person in gallery_dir(original validation set), we split one img of it to query_train set
    for image_name in val_img_names:
        src_path = os.path.join(trainID_dir, image_name)
        tgt_path = os.path.join(valID_dir, image_name)
        os.rename(src_path, tgt_path)

def create_random_split_gallery_query(root_dir):
    def extract_person_id(image_name):
        person_id = image_name.split('_')[0]
        return int(person_id)

    gallery_dir = os.path.join(root_dir, "gallery_train")

    query_dir = os.path.join(root_dir, "queries_train")
    os.makedirs(query_dir, exist_ok=True)

    # split unique image of each distinct person in gallery_train dir, move it to queries_train dir
    images_names = os.listdir(gallery_dir)
    grouped_images = {}
    for img_name in images_names:
        person_id = extract_person_id(img_name)
        if person_id not in grouped_images:
            grouped_images[person_id] = [img_name]
        else:
            grouped_images[person_id].append(img_name)

    unique_img_names = [lyst[random.randint(0,-1+len(lyst))] for _, lyst in grouped_images.items()]
    # print(len(unique_img_names))
    # print(unique_img_names[:10])

    # for each distinct person in gallery_dir(original validation set), we split one img of it to query_train set
    for image_name in unique_img_names:
        src_path = os.path.join(gallery_dir, image_name)
        tgt_path = os.path.join(query_dir, image_name)
        os.rename(src_path, tgt_path)


if __name__ == '__main__':
    root = '../Market'
    # train_path = "../Market_final/train"
    # val_path = "../Market_final/val"
    # annotation_path = "../Market/annotations_train.csv"
    #create_random_split_train_val_attributes(root)
    #print("Finish data preparation :)")
    # create_random_split_gallery_query(root)
    create_random_split_train_val_ID(root)
    print("dataset splitting completed :) ")