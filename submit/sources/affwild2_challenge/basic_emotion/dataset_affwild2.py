import pickle, tqdm
import os, pandas as pd, numpy as np, cv2
import matplotlib.pyplot as plt

emotiw_name  = np.array(["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"])
emotiw_dict  = dict(zip(np.arange(len(emotiw_name)), emotiw_name))
emotiw_idict = dict(zip(emotiw_name, np.arange(len(emotiw_name))))

affwild2_emotiw_mapping = np.array([4, 0, 1, 2, 3, 5, 6])
emotiw_affwild2_mapping = np.array([1, 2, 3, 4, 0, 5, 6])

affwild2_name = np.array(["Neutral", "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"])

affwild2_dict = dict(zip(np.arange(len(affwild2_name)), affwild2_name))
affwild2_dict[-1] = "NotKnown"
affwild2_dict[-2] = "Test"
affwild2_dict[-3] = "NotProcess"
# -3: NotProcess, -2: Test, -1: NotKnown (expr)

# -5: NotKnown, -4: NotProcess, -2: Test (va)

affwild2_idict= dict(zip(affwild2_name, np.arange(len(affwild2_name))))
affwild2_idict["NotKnown"] = -1
affwild2_idict["Test"] = -2
affwild2_idict["NotProcess"] = -3

class AffWild2Dataset:
    def __init__(self, db_file, db_root, **kwargs):
        self.db_file = db_file
        self.db_root = db_root
        
        self.df_frames = pd.read_hdf(self.db_file, key="frames")
        self.df_video  = pd.read_hdf(self.db_file, key="video")
        self.df_emotion= pd.read_hdf(self.db_file, key="emotion")
        # array([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6])
        
        pass
    # __init__
    
    def load_scheme_emotion(self):

        self.a_idx_all   = np.arange(len(self.df_frames))

        self.a_idx_train= self.a_idx_all[np.logical_and(self.df_frames["expr_type"]=="train", self.df_frames["expr_emotion"]>=0)]
        self.n_train    = len(self.a_idx_train)

        self.a_idx_valid= self.a_idx_all[np.logical_and(self.df_frames["expr_type"]=="valid", self.df_frames["expr_emotion"]>=0)]
        self.n_valid    = len(self.a_idx_valid)

        self.a_idx_test = self.a_idx_all[self.df_frames["expr_type"]=="test"]
        self.n_test     = len(self.a_idx_test)

        print("Loading Scheme Emotion: ")
        save_dir = os.path.dirname(self.db_file)
        a_all_scheme = get_index_scheme_emotion(self, has_va=False, save_path=os.path.join(save_dir, "scheme_emotion.pkl"))
        self.a_scheme_train = a_all_scheme["train"]
        self.a_scheme_valid = a_all_scheme["valid"]
        self.a_scheme_test = a_all_scheme["test"]
        pass
    # load_scheme_emotion

    def load_scheme_emotion_va(self):

        self.a_idx_all   = np.arange(len(self.df_frames))

        filter_valaro = np.logical_and(self.df_frames["va_valence"]>=-1, self.df_frames["va_valence"]<=1)
        filter_valaro = np.logical_and(filter_valaro, self.df_frames["va_arousal"]>=-1)
        filter_valaro = np.logical_and(filter_valaro, self.df_frames["va_arousal"]<=1)

        # Train
        filter_train_basic  = np.logical_and(self.df_frames["expr_type"]=="train", self.df_frames["expr_emotion"]>=0)       
        self.a_idx_train    = self.a_idx_all[np.logical_and(filter_train_basic, filter_valaro)]
        self.n_train        = len(self.a_idx_train)

        # Valid
        filter_valid_basic  = np.logical_and(self.df_frames["expr_type"]=="valid", self.df_frames["expr_emotion"]>=0)
        self.a_idx_valid    = self.a_idx_all[np.logical_and(filter_valid_basic, filter_valaro)]
        self.n_valid        = len(self.a_idx_valid)

        # Test Emotion
        self.a_idx_test    = self.a_idx_all[self.df_frames["expr_type"]=="test"]
        self.n_test     = len(self.a_idx_test)

        # Test VA
        self.a_idx_va_test = self.a_idx_all[self.df_frames["va_type"]=="test"]
        self.n_va_test     = len(self.a_idx_va_test)

        print("Loading Scheme Emotion VA: ")   
        save_dir = os.path.dirname(self.db_file)
        a_all_scheme = get_index_scheme_emotion(self, has_va=True, save_path=os.path.join(save_dir, "scheme_emotion_va.pkl"))
        self.a_scheme_train = a_all_scheme["train"]
        self.a_scheme_valid = a_all_scheme["valid"]
        self.a_scheme_test = a_all_scheme["test"]
        self.a_scheme_va_test = a_all_scheme["va_test"]
        pass
    # load_scheme_emotion_va

    def load_scheme_va(self):

        self.a_idx_all   = np.arange(len(self.df_frames))

        filter_valaro = np.logical_and(self.df_frames["va_valence"]>=-1, self.df_frames["va_valence"]<=1)
        filter_valaro = np.logical_and(filter_valaro, self.df_frames["va_arousal"]>=-1)
        filter_valaro = np.logical_and(filter_valaro, self.df_frames["va_arousal"]<=1)

        # Train
        filter_train_basic  = np.logical_and(self.df_frames["va_type"]=="train", filter_valaro)       
        self.a_idx_train    = self.a_idx_all[filter_train_basic]
        self.n_train        = len(self.a_idx_train)

        # Valid
        filter_valid_basic  = np.logical_and(self.df_frames["va_type"]=="valid", filter_valaro)
        self.a_idx_valid    = self.a_idx_all[filter_valid_basic]
        self.n_valid        = len(self.a_idx_valid)

        # Test
        self.a_idx_test    = self.a_idx_all[self.df_frames["va_type"]=="test"]
        self.n_test     = len(self.a_idx_test)
        pass
    # load_scheme_va

    def get_block_idx(self, idx, a_scheme, n_sel = 32):
        idx_sel = a_scheme[self.df_frames.loc[idx]["video_name"]]
        idx_sel = idx_sel[idx_sel <= idx][-n_sel:]
        return idx_sel
    # get_block_idx

    def load_scheme_va_emotion(self):

        self.a_idx_all   = np.arange(len(self.df_frames))

        filter_valaro = np.logical_and(self.df_frames["va_valence"]>=-1, self.df_frames["va_valence"]<=1)
        filter_valaro = np.logical_and(filter_valaro, self.df_frames["va_arousal"]>=-1)
        filter_valaro = np.logical_and(filter_valaro, self.df_frames["va_arousal"]<=1)

        filter_emotion= self.df_frames["expr_emotion"]>=0

        # Train
        filter_train_basic  = np.logical_and(self.df_frames["va_type"]=="train", filter_valaro)       
        self.a_idx_train    = self.a_idx_all[np.logical_and(filter_train_basic,filter_emotion)]
        self.n_train        = len(self.a_idx_train)

        # Valid
        filter_valid_basic  = np.logical_and(self.df_frames["va_type"]=="valid", filter_valaro)
        self.a_idx_valid    = self.a_idx_all[np.logical_and(filter_valid_basic,filter_emotion)]
        self.n_valid        = len(self.a_idx_valid)

        # Test
        self.a_idx_test    = self.a_idx_all[self.df_frames["va_type"]=="test"]
        self.n_test        = len(self.a_idx_test)
        pass
    # load_scheme_va_emotion
      
    def view_images(self, a_idx_ranges, idx_show = None, transforms = None, 
                    expr_show = True, va_show = False, title = "View Images", rows=4, cols=8, figsize=(10, 16), verbose = 1):
        # Choose idx show
        if idx_show is None: 
            idx_show = np.random.choice(len(a_idx_ranges), rows * cols)
        a_idx_show  = a_idx_ranges[idx_show]

        if verbose == 1: print("idx in ranges: ", idx_show)

        if title != "": print(title)
        # View images
        rows = (len(idx_show) + cols - 1) // cols
        for row in range(rows):
            plt.figure(figsize=figsize)
            for col in range(cols):
                idx = row * cols + col
                if idx>=len(idx_show): break
                
                info = self.df_frames.loc[a_idx_show[idx]]

                image_path = self.db_root + "/" + info["path"]
                image  = cv2.imread(image_path)
                
                if transforms is not None:
                    result = transforms(image=image)
                    image  = np.uint8(result["image"])
                # if
                
                plt.subplot(rows, cols, idx + 1), plt.axis("off"), plt.imshow(image[:, :, ::-1])

                s_title = ""
                if expr_show == True:
                    emotion_name = "Unknown"
                    if info["expr_emotion"]>=0 and info["expr_emotion"]<=6:
                        emotion_name = emotiw_name[affwild2_emotiw(info["expr_emotion"])]
                    s_title = emotion_name
                # if
                if va_show == True:
                    va_name = f"{info['va_valence']:.2f} {info['va_arousal']:.2f}"
                    if s_title != "": s_title += "\n"
                    s_title += va_name
                # if
                if s_title != "": plt.title(s_title)
            # for
            plt.show()
        # for
        pass
    # view_images
    
    def view_emotion_summary(self, a_idx_ranges, title = "Data", 
                             title_size = 16, axis_size = 12, axis_roration = 20, figsize=(10, 10)):
        plt.figure(figsize=figsize)
        print("Number of images:\t{:>6}".format(len(a_idx_ranges)))

        y = self.df_frames["expr_emotion"].loc[a_idx_ranges].values

        n_classes = np.unique(y)
        y_bins, y_classes = np.histogram(y, bins=len(n_classes))

        n_name_classes = [affwild2_dict[i] for i in n_classes]

        print("Distribution: ", dict(zip(n_name_classes, y_bins)))

        plt.title("%s Distribution\n" % title, fontsize=16)
        for idx in range(len(n_classes)): 
            plt.bar(affwild2_dict[n_classes[idx]], y_bins[idx])
        plt.xticks(np.arange(len(n_classes)), fontsize=12, rotation=20)
        plt.yticks(fontsize=12, rotation=20)
        plt.show()
        pass
    # view_emotion_summary

    def view_va_summary(self, a_idx_ranges, title = "Data", col_shows = ["va_valence", "va_arousal"], 
                             title_size = 16, axis_size = 12, axis_roration = 20, figsize=(6, 6)):
        print("Number of images:\t{:>6}".format(len(a_idx_ranges)))

        for col in col_shows:
            print("--------------------")
            print(f"Process {col}: ")
            y = self.df_frames[col].loc[a_idx_ranges].values
            print(y)

            y_unknown = y[y<-1]
            if len(y_unknown)>0:
                n_unknown_classes = np.unique(y_unknown)
                y_unknown_bins, _ = np.histogram(y_unknown, bins=len(n_unknown_classes))

                dict_unknown_name = {-5: "NotKnown", -4: "NotProcess", -2: "Test"}
                n_unknown_name_classes = [dict_unknown_name[i] for i in n_unknown_classes]
                print("Invalid distribution: ", dict(zip(n_unknown_name_classes, y_unknown_bins)))
            else:
                print("Invalid distribution: Empty")
            # if

            y_known = y[y>=-1]
            if len(y_known)>0:
                plt.figure(figsize=figsize)
                plt.title("Distribution Histogram of Valid Values\n", fontsize=14)
                plt.hist(y[y>=-1])
                plt.xticks(fontsize=12, rotation=20)
                plt.yticks(fontsize=12, rotation=20)
                plt.show()
            else:
                print("Valid distribution: Empty")
            # if
        # for
        pass
    # view_emotion_summary
# AffWild2Dataset

def affwild2_emotiw(idx):
    return emotiw_idict[affwild2_dict[idx]]
# affwild2_emotiw

def get_index_video(a_idx_data, df_frames):
    a_data = {}
    
    idx_all = np.arange(len(df_frames))
    filter_sel = np.zeros(len(df_frames), np.bool)
    filter_sel[a_idx_data] = True

    video_names = np.unique(df_frames[filter_sel]["video_name"].values)

    for idx_video in tqdm.tqdm_notebook(range(len(video_names)), desc="Process"):
        video_name = video_names[idx_video]
        a_idx_sel = idx_all[np.logical_and(df_frames["video_name"] == video_name, filter_sel).values]
        if len(a_idx_sel)>0:
            a_idx_sel = np.hstack([np.repeat(a_idx_sel[0],32), a_idx_sel, np.repeat(a_idx_sel[-1],32)])
            a_data[video_name] = a_idx_sel
        # if
        pass
    # for
    
    return a_data
# get_index_video

def get_index_scheme_emotion(ds, has_va = False, save_path = "scheme_emotion.pkl"):
    a_scheme = {}
    if os.path.exists(save_path) == False:
        print("Indexing Train: ")
        a_train = get_index_video(ds.a_idx_train, ds.df_frames)
        a_scheme["train"] = a_train

        print("Indexing Valid: ")
        a_valid = get_index_video(ds.a_idx_valid, ds.df_frames)
        a_scheme["valid"] = a_valid

        print("Indexing Test: ")
        a_test = get_index_video(ds.a_idx_test, ds.df_frames)
        a_scheme["test"] = a_test

        if has_va == True:
            print("Indexing TestVA: ")
            a_va_test = get_index_video(ds.a_idx_va_test, ds.df_frames)
            a_scheme["va_test"] = a_va_test
        # if
        
        with open(save_path, "wb") as file:
            pickle.dump(a_scheme, file)
        # with
    else:
        with open(save_path, "rb") as file:
            a_scheme = pickle.load(file)

            for key in a_scheme.keys():
                print(f" + Loading {key}: {len(a_scheme[key])}")
            # for
        # with
    # if
    return a_scheme
# def