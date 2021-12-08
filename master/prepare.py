import os
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

train_path = "../Market/train"
val_path = "../Market/val"
annotation_path = "../Market/annotations_train.csv"

class MarketDataset(Dataset):
  def __init__(self,image_dir, csv_file=None, transform=None, label_transform = T.ToTensor()):
    self.image_dir = image_dir
    self.csv_file = csv_file
    self.transform = transform
    self.label_transform = label_transform
    self.images = os.listdir(image_dir) #list all the files in the folder
    self.csv_df = pd.read_csv(csv_file) if self.csv_file else None

  def __len__(self):
    return len(self.images)

  def __getitem__(self,index):
    #image
    img_path = os.path.join(self.image_dir, self.images[index])
    image = Image.open(img_path).convert("RGB")

    if self.transform is not None:
      image = self.transform(image)
    #process labels if required
    if self.csv_file is None:
      return img_path, image
    else:
      #label
      label_id = int(self.images[index].split('_')[0].lstrip("0"))
      image_label = self.csv_df.loc[self.csv_df['id'] == label_id].values #use list[j][i] where i is img_index and j is col number in csv
      image_label = self.label_transform(image_label)
      return image, image_label, self.images[index]



