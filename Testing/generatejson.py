import SimpleITK as sitk
import os, glob
import json
import numpy as np
import re
from skimage.transform import resize

# read image
def extract_number(filename):
    # Use regular expression to find the number in the filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return -1 
sub_path =  glob.glob('./**/*.nii.gz', recursive=True)
keyword = 'test'
dictout = {keyword:[]}
for i in range (0, len(sub_path)): 
	print (sub_path[i])
	image = sitk.ReadImage(sub_path[i])
	# Get the image array
	image_array = sitk.GetArrayFromImage(image)
	print (image_array.shape)
	re_arr = resize(image_array, (96, 96, 96))
	re_arr = (re_arr - re_arr.min()) / (re_arr.max() - re_arr.min())
	save_name = sub_path[i]

	saved= sitk.GetImageFromArray(re_arr)
	# saved.CopyInformation(img)
	saved.SetOrigin(image.GetOrigin())
	saved.SetSpacing(image.GetSpacing())
	saved.SetDirection(image.GetDirection())
	sitk.WriteImage(saved, save_name)
				
	smalldict = {}
	smalldict['Image'] = './Testing' + sub_path[i][1:]
	if "LungCT" in sub_path[i]:
		smalldict['Tag'] = "Multi"
	else: 
		smalldict['Tag'] = "Single"
	print ('./Training' + sub_path[i][1:])
	dictout[keyword].append(smalldict)

savefilename = './test'+ '.json'
with open(savefilename, 'w') as fp:
	json.dump(dictout, fp)