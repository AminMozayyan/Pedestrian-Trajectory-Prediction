import sys
import os
import os.path as osp
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
import cv2

from lib.utils.data_utils import bbox_denormalize, cxcywh_to_x1y1x2y2
from lib.utils.eval_utils import eval_jaad_pie, eval_jaad_pie_cvae
from lib.losses import cvae, cvae_multi

def train(model, train_gen, criterion, optimizer, device):
    model.train() # Sets the module in training mode.
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _  = model(inputs=input_traj, map_mask=None, targets=target_traj)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            goal_loss = criterion(all_goal_traj, target_traj)

            train_loss = goal_loss + cvae_loss + KLD_loss.mean()

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
    total_goal_loss /= len(train_gen.dataset)
    total_cvae_loss/=len(train_gen.dataset)
    total_KLD_loss/=len(train_gen.dataset)
    
    return total_goal_loss, total_cvae_loss, total_KLD_loss

def val(model, val_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    model.eval()
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(inputs=input_traj, map_mask=None, targets=None,training=False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            

            goal_loss = criterion(all_goal_traj, target_traj)


            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

    val_loss = total_goal_loss/len(val_gen.dataset)\
         + total_cvae_loss/len(val_gen.dataset) + total_KLD_loss/len(val_gen.dataset)
    return val_loss

def test(model, test_gen, criterion, device, original_videos_dir=None):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    MSE_15 = 0
    MSE_05 = 0 
    MSE_10 = 0 
    FMSE = 0 
    FIOU = 0
    CMSE = 0 
    CFMSE = 0
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    
    # Dictionary to cache video captures to avoid reopening videos
    video_cache = {}
    
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(inputs=input_traj, map_mask=None, targets=None, training=False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)

            goal_loss = criterion(all_goal_traj, target_traj)

            test_loss = goal_loss + cvae_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()

            # Visualization output directory
            viz_dir = './viz_last_frame'
            os.makedirs(viz_dir, exist_ok=True)

            # Image resolution used for denormalization (matches configs/jaad/jaad.py)
            W, H = 1920, 1080

            # Draw/save per-sample
            cur_imgs = data.get('cur_image_file', [None] * batch_size)  # list of image paths
            for i in range(batch_size):
                # Reconstruct absolute normalized boxes
                last = input_traj_np[i, -1]                       # (4,)
                obs_abs_norm = input_traj_np[i]                   # (Tobs, 4)
                gt_abs_norm = last[None, :] + target_traj_np[i, -1]      # (Tpred, 4)
                k = 0  # choose a CVAE mode to visualize
                pred_abs_norm = last[None, :] + cvae_dec_traj[i, -1, k]  # (Tpred, 4)

                # Denormalize to pixel coords
                obs_px = bbox_denormalize(obs_abs_norm, W=W, H=H)
                gt_px = bbox_denormalize(gt_abs_norm,   W=W, H=H)
                pred_px = bbox_denormalize(pred_abs_norm, W=W, H=H)

                # Convert to x1y1x2y2 for drawing
                obs_xyxy  = cxcywh_to_x1y1x2y2(obs_px).astype(int)
                gt_xyxy   = cxcywh_to_x1y1x2y2(gt_px).astype(int)
                pred_xyxy = cxcywh_to_x1y1x2y2(pred_px).astype(int)

                # Try to load frame from video clip
                img = None
                if original_videos_dir:
                    # Extract video ID and frame number from the current image path
                    img_path = cur_imgs[i] if isinstance(cur_imgs, list) else cur_imgs
                    if img_path:
                        # Parse the path to get video ID and frame number
                        # Expected format: /path/to/images/video_id/frame_number.png
                        path_parts = img_path.replace('\\', '/').split('/')
                        if len(path_parts) >= 2:
                            video_id = path_parts[-2]  # video_id folder
                            frame_filename = path_parts[-1]  # frame_number.png
                            frame_number = int(frame_filename.split('.')[0])  # frame_number
                            
                            # Construct path to video clip
                            video_path = os.path.join(original_videos_dir, f"{video_id}.mp4")
                            
                            if os.path.exists(video_path):
                                # Check if video is already in cache
                                if video_id not in video_cache:
                                    video_cache[video_id] = cv2.VideoCapture(video_path)
                                    print(f"Opened video: {video_path}")
                                
                                # Extract the specific frame
                                cap = video_cache[video_id]
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                                ret, frame = cap.read()
                                
                                if ret:
                                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    print(f"Extracted frame {frame_number} from video: {video_path}")
                                else:
                                    print(f"Failed to extract frame {frame_number} from video: {video_path}")
                            else:
                                print(f"Video not found: {video_path}")
                
                # Fallback to loading from current image path if video not found
                if img is None:
                    img_path = cur_imgs[i] if isinstance(cur_imgs, list) else cur_imgs
                    if img_path and os.path.exists(img_path):
                        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    else:
                        # Create white canvas as last resort
                        img = (np.ones((H, W, 3), dtype=np.uint8) * 255)

                # Draw observed (green), GT future (red), prediction (blue)
                for (x1,y1,x2,y2) in obs_xyxy:
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                for (x1,y1,x2,y2) in gt_xyxy:
                    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
                for (x1,y1,x2,y2) in pred_xyxy:
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

                # Save arrays and overlay
                np.savez(os.path.join(viz_dir, f'b{batch_idx}_i{i}.npz'),
                        observed=obs_px, gt=gt_px, pred=pred_px, image_path=img_path)
                cv2.imwrite(os.path.join(viz_dir, f'overlay_b{batch_idx}_i{i}.png'),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE, batch_CMSE, batch_CFMSE, batch_FIOU =\
                eval_jaad_pie_cvae(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj[:,-1,:,:,:])
            MSE_15 += batch_MSE_15
            MSE_05 += batch_MSE_05
            MSE_10 += batch_MSE_10
            FMSE += batch_FMSE
            CMSE += batch_CMSE
            CFMSE += batch_CFMSE
            FIOU += batch_FIOU
            

    
    # Clean up video captures
    for cap in video_cache.values():
        cap.release()
    
    MSE_15 /= len(test_gen.dataset)
    MSE_05 /= len(test_gen.dataset)
    MSE_10 /= len(test_gen.dataset)
    FMSE /= len(test_gen.dataset)
    FIOU /= len(test_gen.dataset)
    
    CMSE /= len(test_gen.dataset)
    CFMSE /= len(test_gen.dataset)
    

    test_loss = total_goal_loss/len(test_gen.dataset) \
         + total_cvae_loss/len(test_gen.dataset) + total_KLD_loss/len(test_gen.dataset)
    return test_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE


def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
