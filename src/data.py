from torch.utils.data import Dataset
from PIL import Image
import json
import pandas as pd

class COCODataset(Dataset):
    def __init__(self, root, annotation_path, train=True, transform_image=None, transform_text=None, filter_df=None, take_first=None):
        data = json.load(open(annotation_path))
        self.root = root
        self.annotation_path = annotation_path
        self.transform_image = transform_image
        self.transform_text = transform_text
        self.train = train
        
        # get images
        images = {
            'file_name': [],
            'height': [],
            'width': [],
            'image_id': [],
        }

        for image in data['images']:
            images['file_name'].append(image['file_name'])
            images['height'].append(image['height'])
            images['width'].append(image['width'])
            images['image_id'].append(image['id'])
        images = pd.DataFrame(images)
        
        if train:
            # get annotations
            annotations = {
                'image_id': [],
                'id': [],
                'caption': [],
            }

            for annotation in data['annotations']:
                annotations['image_id'].append(annotation['image_id'])
                annotations['id'].append(annotation['id'])
                annotations['caption'].append(annotation['caption'])
            annotations = pd.DataFrame(annotations)
        
        if train:
            # merge images and annotations
            self.df = annotations.merge(images, on='image_id')
        else:
            self.df = images
        
        # free memory
        del images, annotations, data
        
        if filter_df is not None:
            self.df = filter_df(self.df)

        if take_first is not None:
            self.df = self.df[:take_first]
        
        self.df = self.df.reset_index(drop=True)
            
    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        image = self._load_image(file_name)
        
        if self.transform_image is not None:
            image = self.transform_image(image)
        
        if self.train:
            caption = self.df['caption'][idx]
            if self.transform_text:
                caption = self.transform_text(caption)
            
            return image, caption
        
        return image
    
    def __len__(self):
        return len(self.df)
    
    def _load_image(self, file_name):
        return Image.open(self.root + "/" + file_name).convert('RGB')
    