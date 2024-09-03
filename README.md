# UniMo: Universal Motion Correction For Medical Images without Network Retraining

**Jian Wang, Razieh Faghihpirayesh, Danny Joca, Polina Golland, Ali Gholipour**

## Highlights

- **Motion Correction Framework:** UniMo, a Universal Motion Correction framework using deep neural networks for diverse imaging modalities.
- **One-Time Training:** UniMo requires only one-time training on a single modality and maintains high stability and adaptability across multiple unseen image modalities.
- **Joint Learning Framework:** Integrates multimodal knowledge from both shape and images to improve motion correction accuracy despite image appearance changes.
- **Geometric Deformation Augmenter:** Features a geometric deformation augmenter that enhances global motion correction by addressing local deformations and generating augmented data to improve training.
- **Superior Accuracy:** Demonstrated to surpass existing motion correction methods in accuracy across various datasets with four different image modalities.
- **Significant Impact:** Represents a major advancement in medical imaging, particularly for challenging applications involving wide ranges of motion, such as fetal imaging.
## Run training 

- **Python3 Tracking_trainer.py**
The trained model and intermediate results will be saved in the ./saved_model and ./check_result directories. These directories contain some representative results. All parameters used in the training process are specified in the parameter.yml file.

## Run testing

- **Python3 Tracking_Testing_MultiModality.py.py or Testing_MultiModality.ipynb**
Please check the testing procedure and visualized results in our [jupyter notebook for testing](https://github.com/IntelligentImaging/UNIMO/blob/main/Testing_MultiModality.ipynb).

## Data Description

### Single Modality Test

For motion correction and tracking in the single modality test, we included **240 sequences of 4D EPIs from fMRI time series** of participants who underwent fetal MRI scans (Siemens 3T scanner).
The dataset covers gestational ages from **22.57 to 38.14 weeks** (mean **32.39 weeks**). 
Imaging parameters included:
- **Slice Thickness:** 2 to 3 mm
- **Repetition Time (TR):** 2 to 5.6 seconds (mean 3.1 seconds)
- **Echo Time (TE):** 0.03 to 0.08 seconds (mean 0.04 seconds)
- **Flip Angle (FA):** 90 degrees

All brain scans were resampled to **96³** with a voxel resolution of **3 mm³** and underwent intensity normalization.

### Multiple Modality Tests

For multiple modality tests (in all baselines), we incorporated **three different image modalities**, including segmentation labels from varying organs, CT scans, and T2 MRIs, from publicly released medical image datasets:

1. **CT Scans from the Lung CT Segmentation Challenge (LCTSC)(https://www.cancerimagingarchive.net/collection/lctsc/):**
   - **Dataset:** 60 CT scans
   - **Image Details:** 4DCT or free-breathing CT images (slice thickness of 2.5 to 3 mm) from 60 patients across three institutions, divided into 36 training datasets, 12 off-site test datasets, and 12 live test datasets.
   - **Segmentation Labels:** Esophagus, heart, lungs, and spinal cord. For our study, we specifically extracted the left and right lungs for motion correction.
  
2. **MedMNIST Datasets of Varying Organs (https://medmnist.com/):**
   - **Dataset:** 200 images 
   - **Image Details:** 3D CT scans of the adrenal gland, bone fractures, and 3D Magnetic Resonance Angiography (MRA) scans of blood vessel shapes in the brain, with manually-segmented labels.
   - **Preprocessing:** Applied Gaussian smoothing filter to all binary maps.

3. **Brain Tumor MRI Scans from Brain Tumor Segmentation (BraTS) Challenge (https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1):**
   - **Dataset:** 200 public T1-weighted brain scans of different subjects
   - **Image Details:** 3D brain tumor MRI scans with tumor segmentation labels.

All volumes from the aforementioned datasets were resampled to **96³**, with a voxel resolution of **1 mm³**, and underwent intensity normalization and bias field correction.

