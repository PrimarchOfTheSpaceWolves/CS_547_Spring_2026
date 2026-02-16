import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
import subprocess as sub
import cv2
import numpy as np
import pandas as pd
import A01

base_dir = "assign01"
image_dir = os.path.join(base_dir, "images")
gamma_dir = os.path.join(base_dir, "gamma")

from scipy.stats import wasserstein_distance

def compute_histogram(image):
    counts, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    return counts / np.sum(counts)

def compute_loss_histograms(ground_output, pred_output):
    ground_hist = compute_histogram(ground_output)
    est_hist = compute_histogram(pred_output)
    domain = np.arange(256)
    score = wasserstein_distance(
        u_values=domain,       # Location of pile A
        v_values=domain,       # Location of pile B
        u_weights=est_hist,# Size of pile A
        v_weights=ground_hist  # Size of pile B
    )
    return score
      
def check_grad_gamma_images():    
    # For each filename...
    all_gamma_filenames = os.listdir(gamma_dir)
    all_gamma_filenames.sort()     
    total_hist_loss = 0   
    for gamma_filename in all_gamma_filenames:
        # Load output image
        ground_output = cv2.imread(os.path.join(gamma_dir, gamma_filename))
        ground_output = cv2.cvtColor(ground_output, cv2.COLOR_BGR2GRAY)
        
        # Load original image
        filename = gamma_filename.split("_")[-1]        
        image = cv2.imread(os.path.join(image_dir, filename))                
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Predict the gamma
        pred_gamma = A01.estimate_gamma_exponent(image, ground_output)
        pred_transform = A01.get_gamma_transform(pred_gamma)
        pred_output = A01.apply_intensity_transform(image, pred_transform)
        
        # Compute loss from histograms of images
        hist_loss = compute_loss_histograms(ground_output, pred_output)           
        print("*", gamma_filename, "=", hist_loss)
        total_hist_loss += hist_loss
      
    ave_hist_loss = total_hist_loss / len(all_gamma_filenames)   
    print("AVERAGE HISTOGRAM LOSS:", ave_hist_loss)
         
def main():
    check_grad_gamma_images()
        
if __name__ == '__main__':    
    main()
