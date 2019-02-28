
% This demo script runs the ECO tracker with hand-crafted features on the
% included "Crossing" video.

% Add paths
setup_paths();

% Load video information
video_path = 'sequences/sinMed';
[seq, ground_truth] = load_video_info(video_path);

seq.len = 120;

% Run ECO
results = testing_ECO_HC_test(seq);