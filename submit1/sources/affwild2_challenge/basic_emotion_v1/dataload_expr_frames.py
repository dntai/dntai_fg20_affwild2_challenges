import pickle
import cv2, numpy as np, matplotlib.pyplot as plt
from .dataset_affwild2 import affwild2_emotiw_mapping, emotiw_name

class FramesExprImageBalancedDataLoader(object):
    """
    x_data = {"image": ["path1", ...], "bbox": [bbox1, ...]},
    y_data = None or [label1 (0->6), ...]
    transforms: augumenting image
    mode: train --> take the images with balancing the label (0-> 6)
    mode: valid --> take the images with pre-defined index
    preprocessing_image_fn: preprocessing
    """
    def __init__(self, a_idx_ranges, a_scheme, ds, 
                 n_classes,
                 n_blocks = 32,

                 transforms = None, 
                 mode = "train", 
                 
                 has_balance   = True,
                 balanced_mode = "max",

                 capacity = 0,  
                 preprocessing_image_fn = None,
                 **kwargs):
        self.a_idx_ranges  = a_idx_ranges
        self.ds            = ds
        self.a_scheme      = a_scheme
        
        self.n_data    = len(self.a_idx_ranges)
        self.n_classes = n_classes
        self.n_blocks  = n_blocks
        
        self.mode       = mode
        self.capacity   = capacity
        self.transforms = transforms
        self.preprocessing_image_fn   = preprocessing_image_fn
        
        self.has_balance   = has_balance
        self.balanced_mode = balanced_mode
        if self.mode in ["test", "valid"]: self.has_balance = False

        self.update_x_data()
        self.update_y_data()

        self.update_capacity()
    # __init__

    def update_x_data(self):
        # x_data
        self.x_data = self.a_idx_ranges
    # update_x_data

    def update_y_data(self):
        # y_emotion_data
        # Mapping label from AffWild2 to EmotiW for training/valid
        # (n_class --> Invalid: NotKnow, NotProcess, Test)
        self.emotion_name   = np.hstack([emotiw_name, ["Unknown"]])
        self.y_emotion_data = self.ds.df_frames["expr_emotion"].loc[self.a_idx_ranges].values
        
        idx_valid = self.y_emotion_data>=0
        self.y_emotion_data[idx_valid] = affwild2_emotiw_mapping[self.y_emotion_data[idx_valid]]

        idx_invalid = self.y_emotion_data<0
        self.y_emotion_data[idx_invalid] = self.n_classes

        # Summary array idx by emotion categroy
        self.idx_by_classes = [[] for i in range(self.n_classes + 1)]
        for idx in range(self.n_data):
            self.idx_by_classes[self.y_emotion_data[idx]].append(idx)
        # for
        self.n_by_classes = [len(self.idx_by_classes[k]) for k in range(len(self.idx_by_classes))]
    # update_y_data

    def update_capacity(self):
        # Autodetect capacity with balance classes
        has_update = True if self.capacity == 0 else False
        if has_update == True: self.capacity = self.n_data
        if has_update == True:
            if self.has_balance == True:
                if self.balanced_mode == "max":
                    self.capacity = np.max(self.n_by_classes[:self.n_classes]) * self.n_classes
                elif self.balanced_mode == "avg":
                    self.capacity = int(np.average(self.n_by_classes[:self.n_classes]) + self.n_classes - 1) * self.n_classes
                elif self.balanced_mode == "min":
                    self.capacity = np.min(self.n_by_classes[:self.n_classes]) * self.n_classes
                # if
            # if has_balance
        # if
    # update_capacity

    def __len__(self):
        return self.capacity
    # __len__

    def set_blocks(self, value):
        if value > 0: self.n_blocks = value
    # set_blocks    

    def __getitem__(self, index):
        # Calculate index --> real_index
        if self.has_balance == True and self.mode=="train": # for train
            idx_class = index % self.n_classes
            idx_pos   = (index // self.n_classes)%self.n_by_classes[idx_class]
            self.real_index  = self.idx_by_classes[idx_class][idx_pos]
            # print(idx_emotion, idx_position)
        else: # for valid, test
            self.real_index = index
        # if
        if self.real_index >= self.n_data: self.real_index = self.real_index % self.n_data

        # Load x_data
        # self.org_rel_idx_block = get_block_idx(self.x_data[self.real_index], 
        #                           self.x_data, 
        #                           self.ds.df_frames, 
        #                           n_sel = self.n_blocks, 
        #                           verbose = 0)
        self.org_rel_idx_block = self.ds.get_block_idx(self.x_data[self.real_index], 
                                                       self.a_scheme, 
                                                       n_sel = self.n_blocks)
        a_image_path = self.ds.db_root + "/" + self.ds.df_frames["path"].loc[self.org_rel_idx_block].values
        x = []
        for image_path in a_image_path:
            image = cv2.imread(image_path) # BGR Images
            if self.transforms is not None: # transform
                result = self.transforms(image = image)
                image  = result["image"]
                pass 
            # if       
            if self.preprocessing_image_fn is not None: image = self.preprocessing_image_fn(image)
            x.append(image)
        # for
        x = np.array(x)

        # Load y_data
        y_value = self.y_emotion_data[self.real_index]
        
        y = None
        if self.mode in ["train", "valid"] and y_value>=0 and y_value<self.n_classes:
            y = np.zeros(self.n_classes)
            y[y_value] = 1.0
        # if

        return [x[-1], x], y
    # __getitem__
    
    def view_image(self, idx_show = None, cols = 8):
        if idx_show is None: idx_show = np.random.choice(self.capacity, 1)
        [image,block], label = self[idx_show]
        
        sub_emotion_labels    = affwild2_emotiw_mapping[self.ds.df_frames["expr_emotion"].loc[self.org_rel_idx_block].values]
        sub_va_valence_labels = self.ds.df_frames["va_valence"].loc[self.org_rel_idx_block].values
        sub_va_arousal_labels = self.ds.df_frames["va_arousal"].loc[self.org_rel_idx_block].values
        
        label_name = emotiw_name[np.argmax(label)] if label is not None else "Unknown"
        rows = (len(block) + cols - 1) // cols
        
        print(f"View 3D Images: Emotion = {label_name}")
        for row in range(rows):
            plt.figure(figsize=(10, 10))
            for col in range(cols):
                idx = row * cols + col
                if idx>=len(image): break
                sub_image = block[idx]
                plt.subplot(rows, cols, idx + 1), plt.axis("off"), plt.imshow(sub_image[:, :, ::-1])
                plt.title(f"{emotiw_name[sub_emotion_labels[idx]]}\n{sub_va_arousal_labels[idx]:.2f} {sub_va_valence_labels[idx]:.2f}")
            # for
            plt.show()
        # for
    # view_image

# FramesExprImageBalancedDataLoader

def get_block_idx(idx, a_ranges, df_frames, n_sel = 32, verbose = 1):

    info = df_frames.loc[idx]
    if verbose==1: 
        print(f"\n----------------\nInformation at location {idx}: \n{info}")

    idx_all    = np.arange(len(df_frames["video_name"]))
    filter_sel = np.zeros(len(df_frames), np.bool)
    filter_sel[a_ranges]  = True
    

    filter_frames = np.logical_and(df_frames["video_name"] == info["video_name"], 
                    np.logical_and(df_frames["frame_idx"]<=info["frame_idx"], 
                                   df_frames["frame_idx"]>=info["frame_idx"]-n_sel))

    a_idx_sel = idx_all[np.logical_and(filter_sel, filter_frames)]
    a_idx_sel = a_idx_sel[-n_sel:]
    
    
    if len(a_idx_sel) < n_sel:
        a_idx_sel = np.hstack([np.repeat(a_idx_sel[:1], n_sel - len(a_idx_sel), axis = 0), a_idx_sel])

    if verbose == 1: print(a_idx_sel)
    
    return a_idx_sel
# get_block_idx