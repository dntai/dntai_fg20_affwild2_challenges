# Affective Expression Analysis in-the-wild using Multi-Task Temporal Statistical Deep Learning Model
Challenges: **The First Affective Behavior Analysis in-the-wild (ABAW) Competition** <br/>
Homepage: https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/ <br/>
Team Name: **CNU_ADL** <br/>
Team Members: <br/>
(1) **Nhu-Tai Do**, donhutai@gmail.com <br/>
(2) **Tram-Tran Nguyen Quynh**, tramtran2@gmail.com <br/>
(3) **Soo-Hyung Kim**<br/>
Affiliation: Chonnam National University, South Korea <br/>

## Our paper:
**Affective  Expression  Analysis  in-the-wild  using  Multi-Task  TemporalStatistical  Deep  Learning  Model**<br/>
Link: https://arxiv.org/abs/2002.09120<br/>
<pre>
@article{Do2020,
	archivePrefix = {arXiv},
	arxivId = {2002.09120},
	author = {Do, Nhu-Tai and Kim, Soo-Hyung},
	eprint = {2002.09120},
	file = {:home/pc/Documents/dntai/documents/mendeley/2020 - Do, Kim - Affective Expression Analysis in-the-wild using Multi-Task Temporal Statistical Deep Learning Model.pdf:pdf},
	month = {feb},
	title = {{Affective Expression Analysis in-the-wild using Multi-Task Temporal Statistical Deep Learning Model}},
	url = {http://arxiv.org/abs/2002.09120},
	year = {2020}
}
<pre/>

## How to run
1. Download and setup Anaconda3
2. Run setup_envs.sh to install conda environments with Python 3.7, keras, tensorflow, etc. <br/>
3. Unzip two pandas index files of Aff-Wild2 dataset: affwild2_cropped_aligned_frames_v1.zip and affwild2_cropped_frames_v1.zip in [data/AffWild2/data] folder
4. Download and setup Aff-Wild2 dataset:
   1. Extract annotations.zip and copy 3 folder AU_Set, EXPR_Set, VA_Set to **data/AffWild2/data/annotations** folder
   2. Extract ccropped_aligned.zip to **data/AffWild2/data/cropped_aligned** folder
   3. Extract ccropped.zip and merge batch 1&2 folder to **data/AffWild2/data/cropped** folder
   4. Extract videos.zip and merge batch 1&2 folder to **data/AffWild2/data/cropped_aligned** folder
5. Download weight files and copy to folder **submit1/weights** from https://drive.google.com/drive/folders/1rJB2viPCxw93qFSaga3uqC6OfWMKRHn2?usp=sharing
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
* List Models<br/>
![alt text](https://github.com/dntai/dntai_fg20_affwild2_challenges/blob/master/images/table1.png)

* List Results<br/>
![alt text](https://github.com/dntai/dntai_fg20_affwild2_challenges/blob/master/images/table2.png)

* Fusion Results on Validation: E
xpr. Score = 0.533, Valence-Arousal Score  = 0.5126<br/>

* Submission results: 
Track 1 Valence-Arousal Challenge on Validation: 0.484 (1), 0.534 (2), 0.514 (3), and 0.527 (4) <br/>
Track 2 Basic Emotion Recognition Challenge on Validation: 0.501 (1), 0.492 (2), 0.478 (3), and 0.543 (4)<br/>

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