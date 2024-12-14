import random

from torch.utils.data import Dataset
import torch


class TrioCleanSetOfAdvAnno(Dataset):
    def __init__(self, model, clean_df, dirty_df, masked_df, unique_clean_imgs, indices, shuffle=True, number_of_dirty_images=5):
        self.model = model
        self.clean_df = clean_df
        self.dirty_df = dirty_df
        self.masked_df = masked_df
        self.unique_imgs = unique_clean_imgs
        self.indices = indices
        self.shuffle = shuffle
        self.number_of_dirty_images = number_of_dirty_images

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_id = self.unique_imgs[self.indices[idx]]
        boxes = self.clean_df[self.clean_df.image_id == image_id].values[:, 3:7].astype("float")
        labels = self.clean_df[self.clean_df.image_id == image_id].values[:, -2].astype("int")
        target = {'boxes': torch.tensor(boxes), 'labels': torch.tensor(labels)}

        clean_image_with_sizes = self.model.load_image(self.clean_df[self.clean_df.image_id == image_id].values[0, -1])
        clean_image = clean_image_with_sizes[0]
        org_size = clean_image_with_sizes[1]
        scale_params = clean_image_with_sizes[2]

        target = self.model.load_targets((clean_image.shape[2], clean_image.shape[1]), (org_size[1], org_size[0]),
                                         target, scale_params)

        dirty_images_paths = list(set(self.dirty_df[self.dirty_df.image_id == image_id].values[:, -1]))
        masked_images_paths = list(set(self.masked_df[self.masked_df.image_id == image_id].values[:, -1]))
        dirty_images = [self.model.load_image(dip)[0] for dip in dirty_images_paths]
        masked_images = [self.model.load_image(mip)[0] for mip in masked_images_paths]

        temp = list(zip(dirty_images, masked_images))
        if self.shuffle:
            random.shuffle(temp)

        dirty_images = [img[0] for img in temp][:self.number_of_dirty_images]
        masked_images = [img[1] for img in temp][:self.number_of_dirty_images]

        return clean_image, dirty_images, masked_images, target


class StandardDat(Dataset):
    def __init__(self, model, df, unique_imgs, indices):
        self.model = model
        self.df = df
        self.unique_imgs = unique_imgs
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_name = self.unique_imgs[self.indices[idx]]
        boxes = self.df[self.df.my_id == image_name].values[:, 3:7].astype("float")
        labels = self.df[self.df.my_id == image_name].values[:, -2].astype("int")

        if labels[0] == -1:
            target = {'boxes': torch.zeros((0, 4)), 'labels': torch.tensor([], dtype=torch.int64)}
        else:
            target = {'boxes': torch.tensor(boxes), 'labels': torch.tensor(labels)}

        image_path = self.df[self.df.my_id == image_name].values[0, -1]
        image_with_sizes = self.model.load_image(image_path)
        image = image_with_sizes[0]
        org_size = image_with_sizes[1]
        scale_params = image_with_sizes[2]

        target = self.model.load_targets((image.shape[2], image.shape[1]), (org_size[1], org_size[0]),
                                         target, scale_params)

        return image, target, image_path


def custom_collate(data):
    return data

