import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, label
from skimage.io import imshow, imread
from skimage.morphology import remove_small_objects
import os
import sys
import tifffile
from PIL import Image
from glob import glob
from natsort import natsorted
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import zarr
from numcodecs import Blosc
import json


class Data:
    def __init__(self, 
                 properties, 
                 properties_arguments,  
                 imaging_interval=3,
                 pixel_arrays=False,
                 file_ext="zarr",
                 nd2_fname = None,
                 zarr_fname = None,
                 channel_dict_path = None,
                 FOV_dict_path = None,
                 times_dict_path = None,
                 trenches_path=None,
                 masks_path=None,
                 csv_outpath = None,
                 segmentation_channel="PC",
                 min_size=100,
                 binary_masks=False):
        """
        Initialise the Data object.
        """
        
        ### load in zarr files and paths ###
    
        if nd2_fname:
            self.experiment_name = nd2_fname
        else:
            self.experiment_name = os.getcwd().split(os.path.sep)[-1]
        
        if zarr_fname:
            self.zarr_fname = zarr_fname
        else:
            self.zarr_fname = None
            
        self.file_ext = file_ext
        
        
        if trenches_path:
            self.trenches_path = trenches_path
        else:
            self.trenches_path = os.getcwd() + os.path.sep + "trenches_no_halo.zarr" 
        
        if masks_path:
            self.masks_path = masks_path
        else:
            self.masks_path = os.getcwd() + os.path.sep + "masks.zarr" 
           
        self.trenches = zarr.open(self.trenches_path, mode='r')
        self.masks = zarr.open(self.masks_path, mode='r')
        
        if zarr_fname:
            self.channel_dict_path = os.getcwd() + os.path.sep + "metadata_" + zarr_fname + "_channels_" + self.experiment_name + ".json"
        elif channel_dict_path:
            self.channel_dict_path = channel_dict_path
        else:
            self.channel_dict_path = os.getcwd() + os.path.sep + "metadata_trenches_no_halo_channels_" + self.experiment_name + ".json"
            
        with open(self.channel_dict_path, 'r') as f:
            self.channel_dict = json.load(f)  # get channel list from trench zarr metadata json
        
        if zarr_fname:
            self.FOV_dict_path = os.getcwd() + os.path.sep + "metadata_" + zarr_fname + "_FOVs_" + self.experiment_name + ".json"
        elif FOV_dict_path:
            self.FOV_dict_path = FOV_dict_path
        else:
            self.FOV_dict_path = os.getcwd() + os.path.sep + "metadata_trenches_no_halo_FOVs_" + self.experiment_name + ".json"
            
        with open(self.FOV_dict_path, 'r') as f:
            self.FOV_dict = json.load(f)  # get FOV to trench mapping from trench zarr metadata json
        
        if zarr_fname:
            self.times_dict_path = os.getcwd() + os.path.sep + "metadata_" + zarr_fname + "_times_" + self.experiment_name + ".json"
        elif times_dict_path:
            self.times_dict_path = times_dict_path
        else:
            self.times_dict_path = os.getcwd() + os.path.sep + "metadata_trenches_no_halo_times_" + self.experiment_name + ".json"
            
        with open(self.times_dict_path, 'r') as f:
            self.times_dict = json.load(f)  # get zarr position to timepoint mapping from trench zarr metadata json
        
        with open("metadata_nd2_{}.json".format(self.experiment_name), "r") as f:
            self.nd2_meta = json.load(f)
        
        if imaging_interval:
            self.imaging_interval = imaging_interval  
        else: # get from trench zarr metadata
            self.imaging_interval = self.nd2_meta["imaging_interval_ms"]/1000
            
        self.pixel_arrays = pixel_arrays
        self.min_size = min_size
        self.binary_masks = binary_masks
        
        # parellising at the trench level adds a massive joblib overhead and actually slows down the regionprops
        # best to split the whole job into packages the size of the number of joblib workers 
        # (coarser grained parallelisation adds a smaller overhead)
        self.joblib_workers = 16
        
        for key, value in self.channel_dict.items():
            if value == segmentation_channel:
                self.segmentation_channel = int(key)
                
        self.mask_file_name_base = "xy{}_{}_TR{}_T{}-{}.{}"
        
        # needs rewriting for zarr, probably just to divide trenches into joblib_workers number of packages
        def create_joblib_packages(trenches):
            """
            Packages up the trench file names into a nested list of len(self.joblib_workers).
            Each package contains len(self.trenches)/self.joblib_workers file names.
            This allows for coarse grained multi-threading on the whole dataset.
            
            :param trenches: A list of str of all the trench file names for the experiment.
            :type trenches: class: 'list'
            """
            num_imgs = trenches.shape[0] * trenches.shape[1] * trenches.shape[2]
            package_length = int(num_imgs/self.joblib_workers)
            #indices = list(range(0,num_imgs,package_length))
            
            zarr_indices_dict = dict()
            tr_counter = 0
            t_counter = 0
            ch_counter = 0
            for x in range(num_imgs):
                zarr_indices_dict[x] = [tr_counter, t_counter, ch_counter]
                ch_counter = ch_counter + 1
                if ch_counter >= trenches.shape[2]:
                    ch_counter = 0
                    t_counter = t_counter + 1
                    if t_counter >= trenches.shape[1]:
                        t_counter = 0
                        tr_counter = tr_counter + 1
                    
            packages = np.linspace(0, num_imgs, self.joblib_workers, endpoint=False)
            indices = [int(x) for x in packages]
            indices[0] = 0
            indices[-1] = len(zarr_indices_dict) - (package_length + 1)
            joblib_packages = []
            for count, idx in enumerate(indices):
                if count <= len(indices)-2:
                    package = list(range(idx,indices[count+1]))
                else:
                    package = list(range(idx,idx+package_length+1))
                joblib_packages.append(package)
                
            return joblib_packages, zarr_indices_dict
        
        self.joblib_packages, self.zarr_indices_dict = create_joblib_packages(self.trenches)
        
        # get region properties based on a list of specified properties
        self.properties = properties
        self.properties_arguments = properties_arguments
        if csv_outpath:
            self.trench_csv_path = csv_outpath
        else:
            self.trench_csv_path = "csv_trenches/csv_trenches_{}".format(self.lane_mapping[self.lane])
        
        # create a dictionary to store cell properties for each colour channel
        self.data = [{} for n in self.channel_dict.values()]
        for n in range(len(self.channel_dict.values())):
            for item in properties:
                self.data[n].update({item: np.asarray([])})
            
            
    def __str__(self):

        print("Experiment:", self.experiment_name)
        print("Channels:", self.channel_dict.values())
        print("Number of unique trenches is ", str(self.trenches.shape[0]))
        print("Number of time points is", str(self.trenches.shape[1]))
        
        if os.path.exists(os.getcwd()+os.path.sep+"kymographs"): #if kymograph making built in, could make this self.kymo_path
            print("Kymograph files:", len(os.listdir(os.getcwd()+os.path.sep+"kymographs")))
        else:
            print("No kymograph directory detected")
        
        if os.path.exists(os.getcwd()+os.path.sep+self.trench_csv_path):
            print("Trench property files:", len(os.listdir(os.getcwd()+os.path.sep+self.trench_csv_path)))
        else:
            print("No trench property file directory detected")
    
        return None
    
    def get_trench_id(self, fname, trench_id_dict):
        """
        Return the integer unique trench id associated with the file name
        of a given intensity image (format xy{zfill(3)_FOV}_{channel}_TR{trench}_T{zfill(4)_time}.{file_ext})
        
        :param fname: A string of the image file name (following the above format).
        :type fname: class: 'str'
        :param trench_id_dict: A variable containing the output of the assign_unique_trench function.
        :type trench_id_dict: class: 'dict'
        :return: The integer unique trench id associated with the file name in the parameter fname.
        :rtype: class: 'int'
        """
        test_str = "{}_{}_{}".format(fname.split("_")[0], fname.split("_")[1], fname.split("_")[2])
    
        return trench_id_dict.get(test_str)
    
    def get_time(self, fname, file_ext, minutes=True, imaging_interval=None):
        """
        Returns the integer time point associated with cell, either in minutes or as the discrete time point.
        Assumed file name format: xy{FOV}_{channel}_TR{trench}_T{zfill(4)_time}.{file_ext}
    
        :param fname: A string of the image file name (following the above format).
        :type fname: class: 'str'
        :param imaging_interval: The time between images being taken in minutes.
        :type imaging_interval: class: 'int'
        :return: The time point for the image in minutes.
        :rtype: class: 'int'
        """
        time_str = fname.split("_")[-1]
        time_str = time_str.strip("T."+file_ext)
        time = int(time_str)
        if minutes:
            time = int(time*imaging_interval)
    
        return time
    
    def get_channel(self, fname):
        """
        Returns the string describing the colour channel
        of a given intensity image (format xy{FOV}_{channel}_TR{trench}_T{zfill(4)_time}.{file_ext}).
    
        :param fname: A string of the image file name (following the above format).
        :type fname: class: 'str'
        :return: The colour channel for the image.
        :rtype: class: 'str'
        """
        channel = fname.split("_")[1]
    
        return channel
    
    def get_FOV(self, fname):
        """
        Returns the FOV number
        of a given intensity image (format xy{FOV}_{channel}_TR{trench}_T{zfill(4)_time}.{file_ext}).
    
        :param fname: A string of the image file name (following the above format).
        :type fname: class: 'str'
        :return: The FOV number for the image.
        :rtype: class: 'int'
        """
        FOV = int(fname.split("_")[0].strip("xy"))
    
        return FOV
    
    def get_trench_num(self, fname):
        """
        Returns the trench number in the file name (the trench number in the FOV) as an integer
        of a given intensity image (format xy{FOV}_{channel}_TR{trench}_T{zfill(4)_time}.{file_ext}).
    
        :param fname: A string of the image file name (following the above format).
        :type fname: class: 'str'
        :return: The trench number for the image.
        :rtype: class: 'int'
        """
        trench_num = int(fname.split("_")[2].strip("TR"))
    
        return trench_num
    
    def relabel_masks(self, arr):
        """
        A function which takes randomly numbered masks and returns labels ordered 1 -> n, 
        1 labelling the region closest to the top of the image (the mother cell).
        This is necessary as tiled omnipose images are segmented as a whole image, so each cell
        in the large tiled image has a unique label, rather than labelling starting at 1 with each mother.

        :param arr: An array of integers.
        :type arr: class: 'numpy.ndarray'
        :return: new_arr, with the relabelled masks.
        :rtype: class: 'numpy.ndarray'
        """

        unique_rows = np.unique(arr, axis=0, return_index=True) # get all unique rows in the array and their indices
        rows = list(unique_rows[0])
        idxs = list(unique_rows[1])
        sorted_list = sorted([(idxs[i], rows[i]) for i in range(len(idxs))]) # sort the rows from top of the array to bottom
        labels_seen = [0]   # initialise a list of values which the algorithm has already 'seen'
        conversions = {}    # store the mappings from old array labels to new labels
        conversions[0] = 0   # we need to ensure the zero values remain the same
    
        # iterate through unique rows of the array, finding 
        for idx, row in sorted_list:
            unique_elems = np.isin(row, labels_seen) # see if there are any new values in the row tested

            # if there are any new values, create a new mapping from the old label to the new one
            if False in unique_elems: 
                unique_indices = np.where(unique_elems == False)
                new_values_set = set([row[x] for x in unique_indices[0]])
                new_values = [x for x in new_values_set]
                for x in new_values:
                    conversions[x] = len(labels_seen)
                    labels_seen.append(x)

        # define a dictionary look-up function
        def convert_value(x):
            return conversions.get(x)

        # create a function that is broadcastable to a numpy array
        convert_arr = np.frompyfunc(convert_value, 1, 1)
        new_arr = convert_arr(arr).astype("uint8")

        # include a basic error check
        mask_values = np.unique(arr)
        new_mask_values = np.unique(new_arr)
        assert len(mask_values) == len(new_mask_values), "You have either lost or gained a mask value from somewhere"

        return new_arr
    
    def get_cell_props(self, trenches, masks, index, zarr_indices_dict, properties):
        """
        Returns the region properties specified in this function for label and intensity image pair.
        Objects smaller than a minimum size are unlikely to be cells and are removed from the analysis.

        :param label_img_path: The full path to the label image file (a binary mask).
        :type label_img_path: class: 'str'
        :param intensity_img_path: The full path to the intensity image file you wish measure the properties of.
        :type intensity_img_path: class: 'str'
        :param min_size: The minimum size in pixels of a region to be included in the analysis.
        :type min_size: class: 'int'
        :return: A dictionary of all the cell properties measured in a given image.
        :rtype: class: 'dict'
        """
        # get trench, time and channel
        tr = zarr_indices_dict[index][0]
        t = zarr_indices_dict[index][1]
        ch = zarr_indices_dict[index][2]
        # read label image
        label_img = masks[tr,t,ch,:,:]
        labels = remove_small_objects(label_img, min_size=self.min_size)
        labels = self.relabel_masks(labels) 

        # if masks are not labelled (i.e. are binary masks), label them.
        if self.binary_masks:
            label_img = label_img.astype(bool)
            label_img = remove_small_objects(label_img, min_size=min_size)
            labels = label(label_img, connectivity = 1)

        # read intensity image
        intensity_img = trenches[tr,t,ch,:,:]

        data = regionprops_table(labels, intensity_img, properties=properties)
        
        # extract bounding box if requested
        if "bbox" in properties:
            bboxes = []
            for count, value in enumerate(data["bbox-0"]):
                bbox = intensity_img[value:data["bbox-2"][count], data["bbox-1"][count]:data["bbox-3"][count]]
                bboxes.append(bbox)
            data.update({"bbox": bboxes})
            
        return data
    
    ### Reasoning for parallel processing used ###
    # without pre-allocating memory, it is very difficult (possibly impossible) to parallelise this process
    # this is because parallel threads would be trying to modify the same object: how does each thread know it is updating the most up to date object?
    # the maximum parallelisation would be through creating 16 data objects, having each thread modify one of these objects
    # and then concatenating the resulting objects.
    
    def get_trench_props(self, package, save_data=False):
        """
        
        """
        if self.pixel_arrays:
            np.set_printoptions(threshold=sys.maxsize)
        for count, idx in enumerate(package):
        
            props = self.get_cell_props(self.trenches, self.masks, idx, self.zarr_indices_dict, self.properties_arguments)
            
            csv_str = "TR{}_{}_{}".format(str(self.zarr_indices_dict[idx][0]).zfill(4), 
                                          self.times_dict[str(self.zarr_indices_dict[idx][1])], 
                                          self.channel_dict[str(self.zarr_indices_dict[idx][2])])
            
            props.update({"FOV": np.asarray([self.FOV_dict[str(self.zarr_indices_dict[idx][0])] 
                                                 for x in range(len(props.get("label")))])})
            props.update({"trench": np.asarray(int(self.zarr_indices_dict[idx][0]) 
                                                 for x in range(len(props.get("label"))))})
            
            ##TODO should make this a function which pulls the true time point from the json file
            props.update({"time": np.asarray(self.zarr_indices_dict[idx][1]*self.imaging_interval 
                                                    for x in range(len(props.get("label"))))})
            props.update({"channel": np.asarray([self.channel_dict[str(self.zarr_indices_dict[idx][2])] 
                                                 for x in range(len(props.get("label")))])})
            
            props.update({"identity": np.asarray([csv_str for x in range(len(props.get("label")))])})
            
           
        
            csv_str = "TR{}_{}_{}".format(str(self.zarr_indices_dict[idx][0]).zfill(4), 
                                           self.times_dict[str(self.zarr_indices_dict[idx][1])], 
                                           self.channel_dict[str(self.zarr_indices_dict[idx][2])])
            
            if save_data:
                df = pd.DataFrame(props)
                df.to_csv(path_or_buf = "{}/data_{}.csv".format(self.trench_csv_path, csv_str), compression = None)
        
        return None
    
    def generate_data(self, save_data=False, backend="loky"):
        
        try:
            os.mkdir(self.trench_csv_path.split("/")[0])
        except:
            print("higher csv directory already exists")
            pass
        try:
            os.mkdir(self.trench_csv_path)
        except:
            print("{} directory already exists".format(self.trench_csv_path))
            pass
        
        Parallel(n_jobs=self.joblib_workers, backend=backend)(delayed(self.get_trench_props)(package, save_data=save_data) for package in self.joblib_packages)
        
        # return print options to normal
        if self.pixel_arrays:
            np.set_printoptions(threshold = 1000)
        
        return None
        
        
    def load_trench_csv(self, fname):
        df = pd.read_csv(os.path.join(self.trench_csv_path, fname), usecols = self.properties)
        return df
    
    def create_master_df(self, save_csv=False, out_file="all_data_{}_{}.csv"):
        """
        Takes a folder of trench csvs, and collates them all into a single melted dataframe.
        Optionally saves this dataframe as a csv.
        """
        trench_csvs = natsorted([x.split(os.path.sep)[-1] for x in glob("{}*.csv".format(self.trench_csv_path + os.path.sep))])
        individual_dfs = [self.load_trench_csv(x) for x in tqdm(trench_csvs)]
        all_data = pd.concat(individual_dfs, axis=0, ignore_index=True)
        if save_csv:
            if self.zarr_fname:
                all_data.to_csv(path_or_buf=out_file.format(self.zarr_fname, self.experiment_name), compression=None)
            else:
                all_data.to_csv(path_or_buf=out_file.format("trenches", self.experiment_name), compression=None)
            
        
        return all_data
    
    def create_trench_time_series_df(self, trench_str, save_csv=False, out_file="data_{}.csv"):
        """
        Takes a folder of trench csvs, and collates the time series of a single trench into a dataframe.
        Optionally saves this dataframe as a csv.
        """
        trench_csvs = natsorted([x.split(os.path.sep)[-1] for x in glob("{}*.csv".format(self.trench_csv_path + os.path.sep)) if trench_str in x])
        individual_dfs = [self.load_trench_csv(x) for x in tqdm(trench_csvs)]
        trench_time_series = pd.concat(individual_dfs, axis=0, ignore_index=True)
        if save_csv:
            trench_time_series.to_csv(path_or_buf=out_file.format(trench_str), compression=None)
        
        return trench_time_series