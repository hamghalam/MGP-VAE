import sys
from utils import download, unzip
import sys
import glob
import scipy as sp
import scipy.linalg as la
from scipy.ndimage import imread
from scipy.misc import imresize
import os
import h5py
import numpy as np

# where have been downloaded
#data_dir = sys.argv[1]
out_of_sample = False
complete_data = False

complete_data 
modal = b'flair'
data_dir =  '/data/home/mohammad/crop_MRI_convert_to_JPG/2Dimage/'
def main():

    # 1. download and unzip data
    #download_data(data_dir)

    # 2. load data
    RV = import_data()

    # 3. split train, validation and test
    RV2,RV_out_of_sample = split_data(RV)

    # 4. export
    if out_of_sample:
      out_file = os.path.join(data_dir, "data_mri_out_of_sample.h5")
      fout = h5py.File(out_file, "w")
      for key in RV_out_of_sample.keys():
          fout.create_dataset(key, data=RV_out_of_sample[key])
      fout.close()
    
    else:
      out_file = os.path.join(data_dir, "data_mri.h5")
      fout = h5py.File(out_file, "w")
      for key in RV2.keys():
          fout.create_dataset(key, data=RV2[key])
      fout.close()






def unzip_data():

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fnames = [
        "asian.zip",
        "africanamerican.zip",
        "caucasian.zip",
        "hispanic.zip",
        "multiracial.zip",
    ]

    for fname in fnames:
        print(".. unzipping")
        unzip(os.path.join(data_dir, fname), data_dir)


def import_data(size=128):

    files = []
    #orients = ["00F", "30L", "30R", "45L", "45R", "60L", "60R", "90L", "90R"]
    orients = ["flair", "t1", "t1ce", "t2"]
    for orient in orients:
        _files = glob.glob(os.path.join(data_dir, "*_%s.jpg" % orient))
        files = files + _files
    files = sp.sort(files)

    D1id = []
    D2id = []
    Did = []
    Rid = []
    Y = sp.zeros([len(files), size, size, 3], dtype=sp.uint8)
    for _i, _file in enumerate(files):
        y = imread(_file)
        y = imresize(y, size=[size, size], interp="bilinear")
        Y[_i] = y
        fn = _file.split(".jpg")[0]
        fn = fn.split("/")[-1]
        did1, did2, did3, did4, rid = fn.split("_")
        Did.append(did1 + "_" + did2+'_'+did3 + "_" + did4)
        Rid.append(rid)
    Did = sp.array(Did, dtype="|S100")
    Rid = sp.array(Rid, dtype="|S100")

    RV = {"Y": Y, "Did": Did, "Rid": Rid}
    return RV





def split_data(RV):



   
    out_of_sample_data = {}
    out_of_sample_test = {}
    
    sp.random.seed(0)
    n_train = int(4 * RV["Y"].shape[0] / 5.0)
    
    n_test = int(1 * RV["Y"].shape[0] / 10.0)
    idxs   = sp.random.permutation(RV["Y"].shape[0])
    idxs_train = idxs[:n_train]
    idxs_test  = idxs[n_train : (n_train + n_test)]
    idxs_val   = idxs[(n_train + n_test) :]
    
    n_train_incomplete    = int(idxs_train.shape[0]*0.75) 
    idxs_train_incomplete = idxs[:n_train_incomplete]
    
    n_test_incomplete    = int(idxs_test.shape[0]*0.75)
    idxs_test_incomplete = idxs[:n_test_incomplete]
    
    Itrain = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_train)
    Itrain_incomplete = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_train_incomplete)
    
    
    Itest            = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_test)
    Itest_incomplete = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_test_incomplete)
    Ival             = sp.in1d(sp.arange(RV["Y"].shape[0]), idxs_val)
    
    out = {}
    
    for key in RV.keys():
            if complete_data:
                out["%s_train" % key] = RV[key][Itrain]
                out["%s_val"   % key] = RV[key][Ival]
                out["%s_test"  % key] = RV[key][Itest]
            else:
                out["%s_train" % key] = RV[key][Itrain_incomplete]
                out["%s_val"  % key] = RV[key][Itest_incomplete]
                out["%s_test"  % key] = RV[key][Itest_incomplete]
                out_of_sample_data["%s_val"   % key] = RV[key][Ival]
    
    if out_of_sample:                
        modalities = out['Rid_train']
        single_modality   = (modalities == modal)
        multi_modality    = np.logical_not(single_modality)
        
        for key in ['Y','Did','Rid']:
            out_of_sample_data["%s_train" % key] = out["%s_train" % key][multi_modality]
            out_of_sample_data["%s_test"  % key] = out["%s_train" % key][single_modality]
    return  out,out_of_sample_data


if __name__ == "__main__":

    main()
