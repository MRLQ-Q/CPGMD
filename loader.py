import os
from PIL import Image
from torch.utils.data import Dataset
global_seed = 0

class PatientDataGenerator(Dataset):
    def __init__(self, cfp_folder, oct_folder, batch_size=2,train=True,
                 root_path='/mnt/h/OCT/OCTAP-600',transform=None):
        self.cfp_folder = cfp_folder
        self.oct_folder = oct_folder
        self.batch_size = batch_size
        self.train = train
        self.transform = transform
        self.root_path = root_path
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(cfp_folder) if os.path.isfile(os.path.join(cfp_folder, f))]
    def __getitem__(self, item):
        case = self.filenames[item]
        oct_path = os.path.join(self.root_path, 'FULL', case+'.bmp')
        octa_path = os.path.join(self.root_path, 'OCT', case+'.bmp')

        oct = Image.open(oct_path)

        if self.transform is not None:
            oct = self.transform(oct)
        #OCT 代码
        # oct_list = []
        # temp = os.listdir(oct_path)
        # temp = [i.split('.')[0] for i in temp]
        # temp = list(map(int, temp))
        # temp = sorted(temp)
        # for split in temp:
        #     temp1 = Image.open(os.path.join(oct_path, str(split) + '.bmp'))
        #     # temp1 = np.array(temp1)
        #     if self.transform is not None:
        #         temp1 = self.transform(temp1)
        #     temp1 = np.expand_dims(temp1, axis=0)
        #     oct_list.append(temp1)
        # oct = np.concatenate(oct_list, axis=1)
        octa = Image.open(octa_path)
        if self.transform is not None:
            octa = self.transform(octa)

        return oct, octa

    def __len__(self):
        return len(self.filenames)
