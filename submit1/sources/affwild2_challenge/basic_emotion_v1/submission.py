# common
import os, sys, glob, tqdm, numpy as np, argparse
from math import ceil

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

from scipy.interpolate import interp1d

from .losses import ccc, loss_ccc, ccc_numpy
from .metrics import expr_score, f1_score, concordance_cc2
from .dataset_affwild2 import AffWild2Dataset, emotiw_affwild2_mapping, affwild2_name, affwild2_emotiw_mapping, emotiw_name
from .dataset_affwild2 import emotiw_affwild2_mapping, affwild2_name
from prlab.utils.model_reports import plot_confusion_matrix, print_summary, model_report, buffer_print_string

def submit_expr(load_name, load_type, load_dir, save_dir, a_idx_data, ds, submit_name):
    load_path = os.path.join(load_dir, load_name)
    save_dir  = os.path.join(save_dir, submit_name)
    if os.path.exists(save_dir)==False: os.makedirs(save_dir)
    
    print("================================")
    print(f"Submit [{load_type}] - Expr")
    print(f"+ File: {load_path}")
    print(f"+ Nummber of items: {len(a_idx_data)}")
    print(f"+ Save folder: {save_dir}")
    print("================================")
    
    y_results    = dict(np.load(load_path, allow_pickle=True))
    df_info      = ds.df_frames.loc[a_idx_data]
    video_names  = np.unique(df_info["video_name"])

    # Emotion
    y_pred_emotion_label = emotiw_affwild2_mapping[np.argmax(y_results["emotion"], axis = 1)]
    s_title_emotion      = "Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise"
    print(f"Nummber of frames: {len(y_pred_emotion_label)}")

    for i_video in tqdm.tqdm(range(len(video_names)), desc="Process video"):
        video_name    = video_names[i_video]
        # if "134-30-1280x720" != video_name: continue
        video_path    = os.path.join(save_dir, video_name + ".txt")
        
        df_video      = ds.df_frames.query(f"video_name=='{video_name}'")

        video_pred_filter_idxes  = df_info.query(f"video_name=='{video_name}'").index
        
        video_filter_frame       = df_info["video_name"] == video_name
        video_pred_emotion       = y_pred_emotion_label[video_filter_frame]
        
        df_video.loc[:, "expr_emotion"] = -1
        df_video.loc[video_pred_filter_idxes, "expr_emotion"] = video_pred_emotion
        
#         display.display(video_pred_emotion)
#         display.display(df_video.loc[video_pred_filter_idxes[-1]]["frame_idx"])
#         display.display(df_video["expr_emotion"].values)
        
        first_emotion   = video_pred_emotion[0]
        first_frame_idx = int(df_video.loc[video_pred_filter_idxes[0]]["frame_idx"])
        last_emotion    = video_pred_emotion[-1]
        last_frame_idx  = int(df_video.loc[video_pred_filter_idxes[-1]]["frame_idx"])
        
        prev_emotion    = -1
        
#         print(first_frame_idx, " ", first_emotion)
#         print(last_frame_idx, " ", last_emotion)
        
    
        with open(video_path, "wt") as file:
            file.writelines(f"{s_title_emotion}\n")
            for frame_idx, emotion in df_video[["frame_idx", "expr_emotion"]].values: 
                if emotion == -1:
                    if frame_idx<=first_frame_idx:
                        emotion = first_emotion
                    elif frame_idx>=last_frame_idx:
                        emotion = last_emotion
                    else:
                        emotion = prev_emotion
                    # if
                else: # emotion != -1:
                    prev_emotion = emotion
                # if
                file.writelines(f"{emotion}\n")
            pass
        # with
        
        # return 
    
        # print(video_name, len(df_video))
        # print(len(df_video.query("expr_emotion==-1")))
        # print(len(df_video.query("expr_emotion>=0")))
        # return 
        pass
    # for
# submit_expr

def submit_va(load_name, load_type, load_dir, save_dir, a_idx_data, ds, submit_name):
    load_path = os.path.join(load_dir, load_name)
    save_dir  = os.path.join(save_dir, submit_name)
    if os.path.exists(save_dir)==False: os.makedirs(save_dir)
    
    print("================================")
    print(f"Submit [{load_type}] - Valence-Arousal")
    print(f"+ File: {load_path}")
    print(f"+ Nummber of items: {len(a_idx_data)}")
    print(f"+ Save folder: {save_dir}")
    print("================================")
    
    y_results   = dict(np.load(load_path, allow_pickle=True))
    df_info     = ds.df_frames.loc[a_idx_data]
    video_names = np.unique(df_info["video_name"])

    # Valence-Arousal
    aro_pred   = y_results["aro_ccc"].flatten()
    val_pred   = y_results["val_ccc"].flatten()
    s_title_va = "valence,arousal"
    print(f"Nummber of frames: {len(aro_pred)}")

    for i_video in tqdm.tqdm(range(len(video_names)), desc="Process video"):
        video_name    = video_names[i_video]
        
        # if "video30" != video_name: continue
        
        video_path    = os.path.join(save_dir, video_name + ".txt")
        
        df_video      = ds.df_frames.query(f"video_name=='{video_name}'")
        
        video_pred_filter_idxes  = df_info.query(f"video_name=='{video_name}'").index
        
        # Valence-Arousal
        video_filter_frame   = df_info["video_name"] == video_name
        video_pred_val       = val_pred[video_filter_frame]
        video_pred_aro       = aro_pred[video_filter_frame]
        
        df_video.loc[:, "val_arousal"] = -5
        df_video.loc[:, "val_valence"] = -5
        df_video.loc[video_pred_filter_idxes, "val_arousal"] = video_pred_aro
        df_video.loc[video_pred_filter_idxes, "val_valence"] = video_pred_val
        
        first_val       = video_pred_val[0]
        first_aro       = video_pred_aro[0]
        first_frame_idx = int(df_video.loc[video_pred_filter_idxes[0]]["frame_idx"])
        last_val        = video_pred_val[-1]
        last_aro        = video_pred_aro[-1]
        last_frame_idx  = int(df_video.loc[video_pred_filter_idxes[-1]]["frame_idx"])
        
        prev_val        = -1
        prev_aro        = -1
        
        xx_aro  = df_video.loc[:, "val_arousal"].values
        idx_aro = np.where(xx_aro!=-5)[0]
        ff_aro  = interp1d(idx_aro + 1, xx_aro[idx_aro], kind='linear', fill_value="extrapolate")
        
        xx_val  = df_video.loc[:, "val_valence"].values
        idx_val = np.where(xx_val!=-5)[0]
        ff_val  = interp1d(idx_val + 1, xx_val[idx_val], kind='linear', fill_value="extrapolate")

        # return df_video.loc[:, "val_arousal"].values
        
        with open(video_path, "wt") as file:
            file.writelines(f"{s_title_va}\n")
            for frame_idx, val, aro in df_video[["frame_idx", "val_valence", "val_arousal"]].values: 
                if val == -5:
                    if frame_idx<=first_frame_idx:
                        val = first_val
                    elif frame_idx>=last_frame_idx:
                        val = last_val
                    else:
                        val = ff_val(frame_idx)
                    # if
                else: # val != -5:
                    prev_val = val
                # if
                
                if aro == -5:
                    if frame_idx<=first_frame_idx:
                        aro = first_aro
                    elif frame_idx>=last_frame_idx:
                        aro = last_aro
                    else:
                        aro = ff_aro(frame_idx)
                    # if
                else: # aro != -5:
                    prev_aro = aro
                # if
                
                file.writelines(f"{val:.3f},{aro:.3f}\n")
            pass
        # with
        
        # return
        pass
    # for
# submit_va

def view_summary(load_name, load_type, load_dir, scheme_name, a_idx_data, ds):
    load_path = os.path.join(load_dir, load_name)
    
    print("================================")
    print(f"Summary [{load_type}]")
    print(f"+ File: {load_path}")
    print(f"+ Scheme: {scheme_name}")
    print(f"+ Nummber of items: {len(a_idx_data)}")
    print("================================")

    
    y_results            = dict(np.load(load_path, allow_pickle=True))
    y_pred_emotion_label = emotiw_affwild2_mapping[np.argmax(y_results["emotion"], axis = 1)]
    y_true_emotion_label = ds.df_frames.loc[a_idx_data]["expr_emotion"].values

    print(f"Nummber of frames: {len(y_pred_emotion_label)}")

    a_summary_report = model_report(y_true_emotion_label, y_pred_emotion_label, affwild2_name)
    s_summary_report = buffer_print_string(print_summary, a_summary_report)
    print(s_summary_report)

    f1   = a_summary_report["model_f1_avg_weighted"]
    acc  = a_summary_report["model_acc_all"]
    expr = f1 * 0.66 + acc * 0.33
    print(f"f1: {f1}")
    print(f"acc: {acc}")
    print(f"expr: {expr}")

    plt.figure(figsize=(6,6))
    plot_confusion_matrix(y_true_emotion_label, y_pred_emotion_label, 
                          title='Average accuracy \n ( Accuracy={acc:.2f} )\n',
                          classes = affwild2_name)

    if scheme_name=="emotion_va":
        aro_pred = y_results["aro_ccc"].flatten()
        val_pred = y_results["val_ccc"].flatten()
        aro_true = ds.df_frames.loc[a_idx_data]["va_arousal"].values
        val_true = ds.df_frames.loc[a_idx_data]["va_valence"].values

        aro_ccc1 = concordance_cc2(aro_true, aro_pred)
        aro_ccc2 = ccc_numpy(aro_true, aro_pred)
        val_ccc1 = concordance_cc2(val_true, val_pred)
        val_ccc2 = ccc_numpy(val_true, val_pred)

        avg_ccc1 = (aro_ccc1 + val_ccc1) / 2.0
        avg_ccc2 = (aro_ccc2 + val_ccc2) / 2.0

        print(f"aro: {aro_ccc1}, {aro_ccc2}")
        print(f"val: {val_ccc1}, {val_ccc2}")
        print(f"valaro_avg: {avg_ccc1}, {avg_ccc2}")
    # if
# view_summary