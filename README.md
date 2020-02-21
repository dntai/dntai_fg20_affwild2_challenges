# Affective Expression Analysis in-the-wild using Multi-Task Temporal Statistical Deep Learning Model
Challenges: **The First Affective Behavior Analysis in-the-wild (ABAW) Competition** <br/>
Homepage: https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/ <br/>
Team Name: **CNU_ADL** <br/>
Team Members: <br/>
(1) **Nhu-Tai Do**, donhutai@gmail.com <br/>
(2) **Tram-Tran Nguyen Quynh**, tramtran2@gmail.com <br/>
(3) **Soo-Hyung Kim**<br/>
Affiliation: Chonnam National University, South Korea <br/>

## How to run
1. Run setup_envs.sh to install conda environments with Python 3.7, keras, tensorflow, etc. <br/>
2. Unzip two pandas index files of Aff-Wild2 dataset: affwild2_cropped_aligned_frames.zip and affwild2_cropped_frames.zip in [data/AffWild2/data] folder
3. Download and setup Aff-Wild2 dataset:
   1. Extract annotations.zip and copy 3 folder AU_Set, EXPR_Set, VA_Set to **data/AffWild2/data/annotations** folder
   2. Extract ccropped_aligned.zip to **data/AffWild2/data/cropped_aligned** folder
   3.  Extract ccropped.zip and merge batch 1&2 folder to **data/AffWild2/data/cropped** folder
   4.  Extract videos.zip and merge batch 1&2 folder to **data/AffWild2/data/cropped_aligned** folder
4. Download weight files and copy to folder **submit/weights** from https://drive.google.com/drive/folders/1rJB2viPCxw93qFSaga3uqC6OfWMKRHn2?usp=sharing