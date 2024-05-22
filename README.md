## Mother machine region properties
This code is used to extract region properties from a mother machine experiment. You will need your cell trench images and segmented mask files available as zarr arrays.
The core functionality derives from scikit-image (https://doi.org/10.7717/peerj.453), around which I have written a class to handle mother machine data and output the data as a csv file.
This code is designed to be used as part of a larger pipeline, and some input files correspond to outputs from upstream parts of the pipeline.
