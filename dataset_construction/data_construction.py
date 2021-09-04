#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pysrt
import sys
import os
import os.path as osp
import numpy as np
import cv2
import h5py
import tqdm
import mediapipe as mp

from glob import glob
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip
from facenet_pytorch import MTCNN
from google.protobuf.json_format import MessageToDict

import torch
# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
#     device = 'cpu'
    
device = 'cpu'
    
# print("device is at {}".format(device))

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
mp_face_mesh = mp.solutions.face_mesh
mtcnn = MTCNN(min_face_size=70, keep_all=True, post_process=False, device=device)

def process_video(video_file,
                  truth,
                  start,
                  end,
                  set_fps = 24,
                  static_image_mode=False,
                  max_num_faces=1,
                  min_detection_confidence=0.1,
                  min_tracking_confidence=0.1,
                  min_detection_probability=0.9,
                  image_padding=0.2):
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print('File Error...')
        exit(0)
        
    # frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frameCount = int((end - start) * fps + 2)
    print(f'Total frames: {frameCount}')

    # DEBUG
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_out = cv2.VideoWriter('debug_video.mp4', fourcc, fps, (w, h))

    face_landmarks = np.zeros((frameCount, 468, 3), dtype='float64')
    face_bbox = np.zeros((frameCount, 4), dtype='int64')
    face_valid_frame = np.zeros((frameCount), dtype='int64')
    face_cropped_image = np.zeros((frameCount, 160, 160, 3), dtype='uint8')

    with mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence) as face_mesh:
        
        new_frame_index = np.rint(np.arange(0, frameCount-1, fps / set_fps)).astype('int64')

        for frame_idx in new_frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
            success, image = cap.read()
            if not success:
                print('End of video stream at frame {}'.format(frame_idx))     
                break
            faces, probabilities = mtcnn.detect(image)
            if faces is None or len(faces) > 1:
                # print(f'Frame {frame_idx}: No face or multiple faces detected')
                # video_out.write(image)
                continue

            if probabilities[0] < min_detection_probability:
                # print(f'Frame {frame_idx}: Low probability on face detection')
                # video_out.write(image)
                continue

            x1, y1, x2, y2 = faces[0].astype('int64')
            pad_x = int((x2-x1)*image_padding)
            pad_y = int((y2-y1)*image_padding)
            cropped_image = np.copy(image[max(y1-pad_y, 0):min(y2+pad_y, h), max(x1-pad_x, 0):min(x2+pad_x, w)])

            face_cropped_image[frame_idx, ...] = cv2.resize(cropped_image, (160, 160))

            # DEBUG
            # cv2.rectangle(image,
            #              (max(x1-pad_x-2, 0), max(y1-pad_y-2, 0)),
            #              (min(x2+pad_x+2, w), min(y2+pad_y+2, h)),
            #              (0, 255, 0), thickness=2)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            cropped_image.flags.writeable = False
            results = face_mesh.process(cropped_image)

            # Draw the face mesh annotations on the image.
            # cropped_image.flags.writeable = True
            if results.multi_face_landmarks:
                # Sometimes face_mesh may not find a face...
                t_array = MessageToDict(results.multi_face_landmarks[0])['landmark']
                t_array = np.array([list(idx.values()) for idx in t_array])
                face_landmarks[frame_idx, ...] = t_array
                face_bbox[frame_idx, ...] = faces[0].astype('int64')
                face_valid_frame[frame_idx] = 1

#                 mp_drawing.draw_landmarks(
#                     image=cropped_image,
#                     landmark_list=results.multi_face_landmarks[0],
#                     connections=mp_face_mesh.FACE_CONNECTIONS,
#                     landmark_drawing_spec=drawing_spec,
#                     connection_drawing_spec=drawing_spec)

            # image[max(y1 - pad_y, 0):min(y2 + pad_y, h), max(x1 - pad_x, 0):min(x2 + pad_x, w)] = cropped_image

            # video_out.write(image)

    cap.release()
 
    with h5py.File(video_file.replace('.mp4', '.hdf5'), 'w') as f:
        f.create_dataset('face_landmarks',
                         data=face_landmarks,
                         compression='gzip',
                         compression_opts=9)
        f.create_dataset('face_bbox',
                         data=face_bbox,
                         compression='gzip',
                         compression_opts=9)
        f.create_dataset('face_valid_frame',
                         data=face_valid_frame,
                         compression='gzip',
                         compression_opts=9)
        f.create_dataset('face_cropped_image',
                         data=face_cropped_image,
                         compression='gzip',
                         compression_opts=9)
        f.attrs["title"] = truth

    print('one subtitles complete!')
    
def mkdirs(d):
    """make dir if not exist"""
    if not osp.exists(d):
        os.makedirs(d)
        
def process_whole_video(video_file,srt_file,save_dir):
    mkdirs(save_dir)
    subtitle = pysrt.open(srt_file)
    for i in range(0,len(subtitle)-1):
        if subtitle[i].text not in ['[Music]', '[Applause]', 'Transcriber:', 'Applause', '(Laughter)', '(Applause)', 'Music', '(Music)'] :
            subtitles_current = subtitle[i]
            subtitles_follow = subtitle[i+1]
            start = subtitles_current.start.hours*3600 + subtitles_current.start.minutes * 60 + subtitles_current.start.seconds + subtitles_current.start.milliseconds / 1000
            end = subtitles_follow.start.hours*3600 + subtitles_follow.start.minutes * 60 + subtitles_follow.start.seconds + subtitles_follow.start.milliseconds / 1000
            target_file = osp.join(save_dir, f'cut{i}.mp4')
            ffmpeg_extract_subclip(video_file, start, end, targetname = target_file)
            process_video(target_file,subtitles_current.text,start,end)
            os.remove(target_file)  

    
if __name__ == "__main__":
    index = int(sys.argv[1])
    data_root = '/vol/bitbucket/bh1511/gp_videos'
    # root = "/vol/bitbucket/bh1511/gp_videos_ted"
    save_root = "/vol/bitbucket/mx520/dataset"
    sub_dir_list = glob(osp.join(data_root,"*"))
    sub_dir = sub_dir_list[index]
    if osp.isdir(sub_dir):
        if glob(osp.join(sub_dir,"*English.srt")) and glob(osp.join(sub_dir,"*.mp4")):
            srt_file = glob(osp.join(sub_dir,"*English.srt"))[0]
            video_file = glob(osp.join(sub_dir,"*.mp4"))[0]
            name = sub_dir[sub_dir.rfind('/')+1:] 
            save_dir = osp.join(save_root,name)
            index_1 = video_file.rfind('p_')
            index_2 = video_file.rfind('s')
            index_3 = video_file.rfind('(')
            index_4 = video_file.rfind('-')
            fps = int(video_file[index_1+2:index_2-2])
            quality = int(video_file[index_3+1:index_1])
            model = video_file[index_2+2:index_4]
            if fps > 23 and quality > 479 and model == 'H264':
                process_whole_video(video_file,srt_file,save_dir)
                print("The whole video has been successfully processed")
            else:
                print("The video is discarded due to its bad fps/quality/model")





