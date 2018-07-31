clear; 
clc;
close all;
% Load video information
base_path  = './seq';
video = 'Biker';

video_path = [base_path '/' video];
[seq, ground_truth] = load_video_info(video_path);
seq.VidName = video; 
seq.st_frame = 1; 
seq.en_frame = seq.len; 

% Run SPBACF-main function
learning_rate = 0.013;  % you can use different learning rate.
results       = run_SPBACF(seq, video_path, learning_rate);


    