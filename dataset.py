import glob
from monai import transforms
from torch.utils.data import Dataset


GLOB_PATHS = {
    'ukb': ['*', '*', '*', 'T1_brain_to_MNI.nii.gz'],
    'brats': []

}





def default_transform(spatial_size=(160, 224, 160), scale_ratio=1., minv=0, maxv=1):
    return transforms.Compose([
        transforms.EnsureChannelFirst(),
        transforms.ResizeWithPadOrCrop(spatial_size=spatial_size, mode='minimum'),
        transforms.Orientation(axcodes='RAS'),
        transforms.Spacing(pixdim=scale_ratio),
        transforms.ScaleIntensity(minv=minv, maxv=maxv)
    ])


class DDPMDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        if isinstance(root, list):
            self.input_list = root
        else:
            self.input_list = glob.glob(root)
        self.transform = transform
        self.loader = transforms.LoadImage(image_only=True)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, item):
        x = self.loader(self.input_list[item])
        if self.transform is not None:
            x = self.transform(x)
        return x

