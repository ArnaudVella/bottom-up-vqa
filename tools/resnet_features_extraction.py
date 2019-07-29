from torchvision import transforms
import torchvision
import os
from tqdm import tqdm
import pickle
from PIL import Image
import numpy as np
import torch
import h5py
from torch.autograd import Variable

!rm data/resnet_train_36.hdf5
!rm data/resnet_val_36.hdf5

class Train(object):
    def __init__(self, n_entries, dataroot, enddataroot, features_filename):
        resize = 256

        crop = 224

        _transforms = []
        if resize is not None:
            _transforms.append(transforms.Resize(resize))
        if crop is not None:
            _transforms.append(transforms.CenterCrop(crop))
        _transforms.append(transforms.ToTensor())
        _transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(_transforms)

        print("Loading %s" % features_filename)
        if True: #not os.path.exists(features_filename):
            print("Feature filename", features_filename,"not found, extracting...")
            self.compute_features(n_entries, dataroot,enddataroot, "layer3", features_filename)

    def load_folder(self, folder, suffix):
        imgs = []
        for f in sorted(os.listdir(folder)):
            if f.endswith(suffix):
                imgs.append(os.path.join(folder, f))
        return imgs


            
            
    def load_imageid(self, folder):
      images = self.load_folder(folder, 'jpg')
      img_ids = set()
      for img in images:
          img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
          img_ids.add(img_id)
      return img_ids
  
    def _read_image(self, fname):
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img)


    def compute_features(self, entries, dataroot, enddataroot, layer, features_filename):
        model = torchvision.models.resnet18(pretrained=True).cuda()
        model.eval()
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        getattr(model, layer).register_forward_hook(get_activation(layer))

        trainroot = os.path.join(dataroot, enddataroot)
        train_set = set(self.load_imageid(trainroot))
        print(train_set)
        #valroot = os.path.join(dataroot, "val2014")
        #val_set = set(self.load_imageid(valroot))
        if entries == []:
            entries = train_set#.union(val_set)

        self.features = {}
        #print(entries)
        mon_dataset=None
        i=0
        for id in tqdm(entries):
            if id in self.features:
               continue

            if id in train_set:
                filename = os.path.join(trainroot, 'COCO_%s_%012d.jpg' % (enddataroot,id))
            else:
                filename = os.path.join(valroot, 'COCO_%s_%012d.jpg' % ("val2014",id))

            assert os.path.exists(filename), filename + " does not exists"
            image = self._read_image(filename)
            inp = torch.from_numpy(image.numpy())
            
            inp = torch.unsqueeze(inp, 0).cuda() # 1,dim,x,x
            inp = Variable(inp)
            
            model(inp)
            out = activation[layer].squeeze(0)
            out = out.view(out.size(0), -1)
            out = out.permute(1,0)
            self.features[id] = out.cpu().numpy().astype(np.float16)
            #self.features[id] = out.cpu()

            if mon_dataset==None:
           
              mon_dataset=not None
              h5_file = h5py.File(features_filename, 'a')
              h5_file.flush()
              shape= len(entries), out.size()[0], out.size()[1]
              print(shape)
              image_features = h5_file.create_dataset(name='image_features', shape=shape, dtype="float16")
              #spatial_features = h5_file.create_dataset(name='spatial_features', shape=shape, dtype="float16")
              #spatial_features [:,:,:]=1
            image_features[i, :,:] = out.cpu()
            #print(type(out.cpu()))
            #image_features[i, :,:] = (Variable(out).data).cpu().numpy().astype(np.float16)
            i=i+1

            
            

        print("Dumping in filename %s with size %d" % (features_filename, len(self.features)))
        pickle.dump(self.features, open(features_filename,'wb+'))
        del model
        h5_file.close()


Train([], "data", "train2014", "data/resnet_train_36.hdf5")
Train([], "data", "val2014", "data/resnet_val_36.hdf5")
