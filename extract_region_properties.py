from data_zarr import *

# choose which properties to extract
properties_regionprops = ["area", 
                          "major_axis_length", 
                          "minor_axis_length", 
                          "centroid-0", 
                          "centroid-1",
                          "orientation",
                          "mean_intensity", 
                          "max_intensity", 
                          "min_intensity", 
                          "label"] 

properties_manual = ["FOV", "trench", "time", "channel", "identity"]
properties = properties_regionprops + properties_manual

properties_arguments = ["area", 
                        "major_axis_length", 
                        "minor_axis_length", 
                        "centroid",
                        "orientation",
                        "mean_intensity", 
                        "max_intensity", 
                        "min_intensity", 
                        "label"] 

imaging_interval = 0.5 # imaging interval in minutes

### update file names below before running ###
d = Data(properties, 
         properties_arguments,  
         imaging_interval=imaging_interval,
         pixel_arrays=False,
         file_ext="zarr",
         nd2_fname = "20240503_SB8_WT_sytox_phage",
         zarr_fname = "trenches_no_halo",
         channel_dict_path = None,
         FOV_dict_path = None,
         times_dict_path = None,
         trenches_path="trenches_no_halo.zarr",
         masks_path="masks.zarr",
         csv_outpath = "csv_trenches",
         segmentation_channel="PC",
         min_size=80,
         binary_masks=False)

# generate individual trench region property csvs
d.generate_data(save_data=True)

# concatenate all region properties into a single dataframe to be saved as a csv file
df = d.create_master_df(save_csv=True)