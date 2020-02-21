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
1. Download and setup Anaconda3
2. Run setup_envs.sh to install conda environments with Python 3.7, keras, tensorflow, etc. <br/>
3. Unzip two pandas index files of Aff-Wild2 dataset: affwild2_cropped_aligned_frames.zip and affwild2_cropped_frames.zip in [data/AffWild2/data] folder
4. Download and setup Aff-Wild2 dataset:
   1. Extract annotations.zip and copy 3 folder AU_Set, EXPR_Set, VA_Set to **data/AffWild2/data/annotations** folder
   2. Extract ccropped_aligned.zip to **data/AffWild2/data/cropped_aligned** folder
   3. Extract ccropped.zip and merge batch 1&2 folder to **data/AffWild2/data/cropped** folder
   4. Extract videos.zip and merge batch 1&2 folder to **data/AffWild2/data/cropped_aligned** folder
5. Download weight files and copy to folder **submit/weights** from https://drive.google.com/drive/folders/1rJB2viPCxw93qFSaga3uqC6OfWMKRHn2?usp=sharing
6. Open JupyterLab and run *.ipynb in **submit** folder to output the results (
   + Run sel_t[xx].ipynb to output the prediction files(modify params parameter if neccessary)
   + Run sel_t[xx]_submit.ipynb to output the result folder (modify params parameter if neccessary)
   
## Proposed model
![alt text](https://github.com/dntai/dntai_fg20_affwild2_challenges/blob/master/images/fig1_model.png)

## Aff-Wild2 dataset
* Overview cropped_aligned image frames in different videos
![alt text](https://github.com/dntai/dntai_fg20_affwild2_challenges/blob/master/images/fig2_problem.png)

* Overview cropped_aligned image frames in the same videos
![alt text](https://github.com/dntai/dntai_fg20_affwild2_challenges/blob/master/images/fig3_affwild2_images.png)

* Data Distribution in Basic Emotion Recognition Track on Training and Validation
![alt text](https://github.com/dntai/dntai_fg20_affwild2_challenges/blob/master/images/distribution_train_valence_arousal.png)

* Data Distribution in Valence-Arousal Regression Track on Training and Validation
![alt text](https://github.com/dntai/dntai_fg20_affwild2_challenges/blob/master/images/distribution_valid_valence_arousal.png)

## Result
* List Models
![alt text](https://github.com/dntai/dntai_fg20_affwild2_challenges/blob/master/images/table1.png)

* List Results
![alt text](https://github.com/dntai/dntai_fg20_affwild2_challenges/blob/master/images/table2.png)

Baseline paper: <br/>
<pre>
@misc{kollias2020analysing,
    title={Analysing Affective Behavior in the First ABAW 2020 Competition},
    author={Dimitrios Kollias and Attila Schulc and Elnar Hajiyev and Stefanos Zafeiriou},
    year={2020},
    eprint={2001.11409},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
<pre/>