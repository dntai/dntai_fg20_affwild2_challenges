import cv2, numpy as np, matplotlib.pyplot as plt
from .dataset_affwild2 import affwild2_emotiw_mapping, emotiw_name

class VAExprImageBalancedDataLoader(object):
    """
    x_data = {"image": ["path1", ...], "bbox": [bbox1, ...]},
    y_data = None or [label1 (0->6), ...]
    transforms: augumenting image
    mode: train --> take the images with balancing the label (0-> 6)
    mode: valid --> take the images with pre-defined index
    preprocessing_image_fn: preprocessing
    """
    def __init__(self, a_idx_ranges, ds, 
                 n_classes,

                 transforms = None, 
                 mode = "train", 
                 
                 has_balance   = True,
                 balanced_mode = "max",

                 capacity = 0,  
                 preprocessing_image_fn = None,
                 **kwargs):
        self.a_idx_ranges  = a_idx_ranges
        self.ds            = ds
        
        self.n_data    = len(self.a_idx_ranges)
        self.n_classes = n_classes
        
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
        self.x_data = self.ds.df_frames["path"].loc[self.a_idx_ranges].values
        self.x_data = self.ds.db_root + "/" + self.x_data
    # update_x_data

    def update_y_data(self):
        # y_emotion_data
        # Mapping label from AffWild2 to EmotiW for training/valid
        # (n_class --> Invalid: NotKnow, NotProcess, Test)
        self.emotion_name   = np.hstack([emotiw_name, ["Unknown"]])
        self.y_emotion_data = self.ds.df_frames["expr_emotion"].loc[self.a_idx_ranges].values
        self.y_va_arousal_data = self.ds.df_frames["va_arousal"].loc[self.a_idx_ranges].values
        self.y_va_valence_data = self.ds.df_frames["va_valence"].loc[self.a_idx_ranges].values
        
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
        image_path = self.x_data[self.real_index]
        x = cv2.imread(image_path) # BGR Images
        
        if self.transforms is not None: # transform
            result = self.transforms(image = x)
            x  = result["image"]
        # if       
        if self.preprocessing_image_fn is not None: x = self.preprocessing_image_fn(x)

        # Load y_data
        y_value = self.y_emotion_data[self.real_index]
        y_va_arousal = self.y_va_arousal_data[self.real_index]
        y_va_valence = self.y_va_valence_data[self.real_index]
        
        y = None
        if self.mode in ["train", "valid"] and y_value>=0 and y_value<self.n_classes:
            y = np.zeros(self.n_classes)
            y[y_value] = 1.0
        # if

        return x, [y, y_va_arousal, y_va_valence]
    # __getitem__
    
    def view_images(self, idx_show = None, rows=4, cols=8):
        if idx_show is None: idx_show = np.random.choice(self.capacity, rows * cols)
        rows = (len(idx_show) + cols - 1) // cols

        for row in range(rows):
            plt.figure(figsize=(10, 10))
            for col in range(cols):
                idx = row * cols + col
                if idx>=len(idx_show): break
                image, label = self[idx_show[idx]]
                plt.subplot(rows, cols, idx + 1), plt.axis("off"), plt.imshow(image[:, :, ::-1])
                y, y_va_arousal, y_va_valence = label
                if label is None:
                    plt.title("Unknown")
                else:
                    y = np.argmax(y)
                    plt.title(f"{emotiw_name[y]}\n{y_va_arousal:.2f} {y_va_valence:.2f}")
            # for
            plt.show()
        # for
    # view_images
# VAExprImageBalancedDataLoader