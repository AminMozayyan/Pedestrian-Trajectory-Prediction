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

import lib.utils as utl
from configs.jaad import parse_sgnet_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.jaadpie_train_utils_cvae import train, val, test


class KaggleModelSaver:
    
    def __init__(self, output_dir, model_name="trajectory_model"):
        self.output_dir = output_dir
        self.model_name = model_name
        self.best_metric = float('inf')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False, is_final=False):
        """Save checkpoint with automatic naming"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        # Save latest
        latest_path = os.path.join(self.output_dir, f"{self.model_name}_latest.pth")
        torch.save(checkpoint, latest_path)
        
        # Save epoch-specific (every 5 epochs to save space)
        if epoch % 5 == 0 or epoch == 1:
            epoch_path = os.path.join(self.output_dir, f"{self.model_name}_epoch_{epoch:03d}.pth")
            torch.save(checkpoint, epoch_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, f"{self.model_name}_best.pth")
            torch.save(checkpoint, best_path)
            self.best_metric = metrics.get('val_loss', float('inf'))
            print(f"Best model saved: {best_path}")
        
        # Save final model
        if is_final:
            final_path = os.path.join(self.output_dir, f"{self.model_name}_final.pth")
            torch.save(checkpoint, final_path)
            print(f"Final model saved: {final_path}")
            
            # Also save weights-only version for inference
            weights_path = os.path.join(self.output_dir, f"{self.model_name}_weights_final.pth")
            torch.save(model.state_dict(), weights_path)
            print(f"Final weights saved: {weights_path}")
        
        print(f"Checkpoint saved: {latest_path}")
    
    def cleanup_old_checkpoints(self, keep_last_n=3):
        """Remove old checkpoints to save space"""
        checkpoint_files = [f for f in os.listdir(self.output_dir) 
                          if f.startswith(f"{self.model_name}_epoch_") and f.endswith('.pth')]
        
        if len(checkpoint_files) > keep_last_n:
            # Sort by epoch number
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # Remove oldest checkpoints
            for old_file in checkpoint_files[:-keep_last_n]:
                old_path = os.path.join(self.output_dir, old_file)
                os.remove(old_path)
                print(f"Removed old checkpoint: {old_file}")

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    
    # Kaggle-optimized saving paths
    if os.path.exists('/kaggle/working'):
        # Running in Kaggle - save to working directory for easy download
        save_dir = '/kaggle/working'
        print("Running in Kaggle - saving to /kaggle/working/")
    else:
        # Running locally - use original path
        save_dir = osp.join(this_dir, 'checkpoints', model_name, str(args.seed))
        print(f"💻 Running locally - saving to {save_dir}")
    
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))


    model = build_model(args)
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                            min_lr=1e-10)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']

    criterion = rmse_loss().to(device)

    train_gen = utl.build_data_loader(args, 'train')
    val_gen = utl.build_data_loader(args, 'val')
    test_gen = utl.build_data_loader(args, 'test')
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())



    # Initialize Kaggle Model Saver
    saver = KaggleModelSaver(output_dir=save_dir, model_name=model_name)
    
    # train
    min_loss = 1e6
    min_MSE_15 = 10e5
    best_model = None
    best_model_metric = None

    for epoch in range(args.start_epoch, args.epochs+args.start_epoch):
        print("Number of training samples:", len(train_gen))

        # train
        train_goal_loss, train_cvae_loss, train_KLD_loss = train(model, train_gen, criterion, optimizer, device)
        # print('Train Epoch: ', epoch, 'Goal loss: ', train_goal_loss, 'Decoder loss: ', train_dec_loss, 'CVAE loss: ', train_cvae_loss, \
        #     'KLD loss: ', train_KLD_loss, 'Total: ', total_train_loss) 
        print('Train Epoch: {} \t Goal loss: {:.4f}\t CVAE loss: {:.4f}\t KLD loss: {:.4f}'.format(
                epoch, train_goal_loss, train_cvae_loss, train_KLD_loss))


        # val
        val_loss = val(model, val_gen, criterion, device)
        lr_scheduler.step(val_loss)


        # test
        test_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE = test(model, test_gen, criterion, device, args.original_videos_dir)
        print("Test Loss: {:.4f}".format(test_loss))
        print("MSE_05: %4f;  MSE_10: %4f;  MSE_15: %4f\n" % (MSE_05, MSE_10, MSE_15))

        # Prepare metrics for saving
        metrics = {
            'train_goal_loss': train_goal_loss,
            'train_cvae_loss': train_cvae_loss,
            'train_KLD_loss': train_KLD_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'MSE_15': MSE_15,
            'MSE_05': MSE_05,
            'MSE_10': MSE_10,
        }
        
        # Determine if this is the best model
        is_best = val_loss < min_loss
        is_final = epoch == (args.epochs + args.start_epoch - 1)
        
        # Save checkpoint using KaggleModelSaver
        saver.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best,
            is_final=is_final
        )
        
        # Update best metrics
        if val_loss < min_loss:
            min_loss = val_loss
            print(f"New best validation loss: {val_loss:.4f}")
        
        if MSE_15 < min_MSE_15:
            min_MSE_15 = MSE_15
            print(f"New best MSE_15: {MSE_15:.4f}")
        
        # Clean up old checkpoints every 10 epochs
        if epoch % 10 == 0:
            saver.cleanup_old_checkpoints(keep_last_n=3)



if __name__ == '__main__':
    main(parse_args())
