function results = testing_ECO_HC(seq, res_path, bSaveImage, parameters)

% Feature specific parameters
hog_params.cell_size = 6;
hog_params.compressed_dim = 10;

grayscale_params.colorspace='gray';
grayscale_params.cell_size = 1;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.compressed_dim = 3;

ic_params.tablename = 'intensityChannelNorm6';
ic_params.useForColor = false;
ic_params.cell_size = 4;
ic_params.compressed_dim = 3;

% Which features to include
params.t_features = {
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    struct('getFeature',@get_table_feature, 'fparams',ic_params),...
};

% Global feature parameters1s
params.t_global.normalize_power = 2;    % Lp normalization with this p
params.t_global.normalize_size = true;  % Also normalize with respect to the spatial size of the feature
params.t_global.normalize_dim = true;   % Also normalize with respect to the dimensionality of the feature

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 4.0;         % The scaling of the target size to get the search area
params.min_image_sample_size = 150^2;   % Minimum area of image samples
params.max_image_sample_size = 200^2;   % Maximum area of image samples

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

% Learning parameters
params.output_sigma_factor = 1/16;		% Label function sigma
params.learning_rate = 0.009;	 	 	% Learning rate
params.nSamples = 30;                   % Maximum number of stored training samples
params.sample_replace_strategy = 'lowest_prior';    % Which sample to replace when the memory is full
params.lt_size = 0;                     % The size of the long-term memory (where all samples have equal weight)
params.train_gap = 5;                   % The number of intermediate frames with no training (0 corresponds to training every frame)
params.skip_after_frame = 10;           % After which frame number the sparse update scheme should start (1 is directly)
params.use_detection_sample = true;     % Use the sample that was extracted at the detection stage also for learning

% Factorized convolution parameters
params.use_projection_matrix = true;    % Use projection matrix, i.e. use the factorized convolution formulation
params.update_projection_matrix = true; % Whether the projection matrix should be optimized or not
params.proj_init_method = 'pca';        % Method for initializing the projection matrix
params.projection_reg = 1e-7;	 	 	% Regularization paremeter of the projection matrix

% Generative sample space model parameters
params.use_sample_merge = true;                 % Use the generative sample space model to merge samples
params.sample_merge_type = 'Merge';             % Strategy for updating the samples
params.distance_matrix_update_type = 'exact';   % Strategy for updating the distance matrix

% Conjugate Gradient parameters
params.CG_iter = 5;                     % The number of Conjugate Gradient iterations in each update after the first frame
params.init_CG_iter = 10*15;            % The total number of Conjugate Gradient iterations used in the first frame
params.init_GN_iter = 10;               % The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
params.CG_use_FR = false;               % Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
params.CG_standard_alpha = true;        % Use the standard formula for computing the step length in Conjugate Gradient
params.CG_forgetting_rate = 50;	 	 	% Forgetting rate of the last conjugate direction
params.precond_data_param = 0.75;       % Weight of the data term in the preconditioner
params.precond_reg_param = 0.25;	 	% Weight of the regularization term in the preconditioner
params.precond_proj_param = 40;	 	 	% Weight of the projection matrix part in the preconditioner

% Regularization window parameters
params.use_reg_window = true;           % Use spatial regularization or not
params.reg_window_min = 1e-4;			% The minimum value of the regularization window
params.reg_window_edge = 10e-3;         % The impact of the spatial regularization
params.reg_window_power = 2;            % The degree of the polynomial to use (e.g. 2 is a quadratic window)
params.reg_sparsity_threshold = 0.05;   % A relative threshold of which DFT coefficients that should be set to zero

% Interpolation parameters
params.interpolation_method = 'bicubic';    % The kind of interpolation kernel
params.interpolation_bicubic_a = -0.75;     % The parameter for the bicubic interpolation kernel
params.interpolation_centering = true;      % Center the kernel at the feature sample
params.interpolation_windowing = false;     % Do additional windowing on the Fourier coefficients of the kernel

% Scale parameters for the translation model
% Only used if: params.use_scale_filter = false
params.number_of_scales = 7;            % Number of scales to run the detector
params.scale_step = 1.01;               % The scale factor

% Scale filter parameters
% Only used if: params.use_scale_filter = true
params.use_scale_filter = true;         % Use the fDSST scale filter or not (for speed)
params.scale_sigma_factor = 1/16;       % Scale label function sigma
params.scale_learning_rate = 0.025;		% Scale filter learning rate
params.number_of_scales_filter = 17;    % Number of scales
params.number_of_interp_scales = 33;    % Number of interpolated scales
params.scale_model_factor = 1.0;        % Scaling of the scale model
params.scale_step_filter = 1.02;        % The scale factor for the scale filter
params.scale_model_max_area = 32*16;    % Maximume area for the scale sample patch
params.scale_feature = 'HOG4';          % Features for the scale filter (only HOG4 supported)
params.s_num_compressed_dim = 'MAX';    % Number of compressed feature dimensions in the scale filter
params.lambda = 1e-2;					% Scale filter regularization
params.do_poly_interp = true;           % Do 2nd order polynomial interpolation to obtain more accurate scale

% Visualization
params.visualization = 1;               % Visualiza tracking and detection scores
params.debug = 0;                       % Do full debug visualization

% GPU
params.use_gpu = false;                 % Enable GPU or not
params.gpu_id = [];                     % Set the GPU id, or leave empty to use default

% Initialize
params.seq = seq;

% Run tracker
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    return;
end

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

box_size = 10;
number_of_rects = 10;
min_x = 10;
max_x = 180;
min_y = 50;
max_y = 200;
init_rects = zeros(number_of_rects, 4);
for i=1:number_of_rects
    x = min_x + (max_x-min_x)*rand();
    y = min_y + (max_y-min_y)*rand();
    init_rects(i,:) = [y,x,box_size,box_size];
end
%init_rects = [...
%    [100,20,box_size,box_size];[150,20,box_size,box_size];...
%    [50,50,box_size,box_size];[100,50,box_size,box_size];[150,50,box_size,box_size];[200,50,box_size,box_size];...
%    [50,100,box_size,box_size];[100,100,box_size,box_size];[150,100,box_size,box_size];[200,100,box_size,box_size];...
%    [100,150,box_size,box_size];[150,150,box_size,box_size]];
org_params = params;
org_seq = seq;
clear seq;
clear params;
close all;

for i=1:size(init_rects,1)
    new_seq = org_seq;
    new_seq.init_rect = init_rects(i,:); % [left right corner y,left right corner x,width,heigth]
    new_seq.init_sz = [new_seq.init_rect(1,4), new_seq.init_rect(1,3)];
    new_seq.init_pos = [new_seq.init_rect(1,2), new_seq.init_rect(1,1)] + (new_seq.init_sz - 1)/2;
    [runparams(i), params(i)] = init_tracker(org_params, new_seq, im, is_color_image);
end
org_seq.time = 0;
[seq, im] = get_sequence_frame(org_seq);

scaleFactor = 10;
params(1).visualization = true;
diffpos = zeros(200,size(init_rects,1),2);
if params(1).visualization
    fig_handle = figure('Name', 'Tracking');
end
while true
    fprintf('Frame %d\n', seq.frame);
    newPos = zeros(size(init_rects,1), 2);
    rect_position_vis = zeros(size(init_rects,1),4);
    stepStart = tic;
    oldPos = zeros(size(init_rects,1), 2);
    parfor i=1:size(init_rects,1)
        oldPos(i,:) = runparams(i).pos;
        [runparams(i)] = step(params(i), seq, im, runparams(i));
        newPos(i,:) = runparams(i).pos;
        rect_position_vis(i,:) = [runparams(i).pos([2,1]) - (runparams(i).target_sz([2,1]) - 1)/2, runparams(i).target_sz([2,1])];
        %fprintf('Pos X: %f, Y: %f\n',newPos(i,2),newPos(i,1))
        %fprintf('Target size X %f, Y: %f\n',runparams(i).target_sz(2),runparams(i).target_sz(1))
    end
    diffpos(seq.frame,:,:) = newPos - oldPos;
    fprintf('\t Step time: %f\n ', toc(stepStart));
    
    % visualization
    if params(1).visualization
        plotStart = tic;
        X_old = oldPos(:,2);
        Y_old = oldPos(:,1);
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        
        if seq.frame == 1 %first frame, create GUI
            
        else
            % Do visualization of the sampled confidence scores overlayed
            figure(fig_handle);
        end
        
        imagesc(im_to_show);
        hold on;
        for i=1:size(init_rects,1)
            rectangle('Position',rect_position_vis(i,:), 'EdgeColor','g', 'LineWidth',2);
            quiver(X_old(i),Y_old(i),diffpos(seq.frame,i,2)*scaleFactor,diffpos(seq.frame,i,1)*scaleFactor, 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
        end
        hold off;
        drawnow
        fprintf('\t Plot time: %f\n ', toc(plotStart));
    end
    loadSeqStart = tic;
    [seq, im] = get_sequence_frame(seq);
    % Read image
    if isempty(im)
        break;
    end
    if size(im,3) > 1 && is_color_image == false
        im = im(:,:,1);
    end
    fprintf('\t Load seq time: %f\n ', toc(loadSeqStart));
end
results = runparams.results;

x = linspace(1,seq.frame-1,seq.frame-1);
for i=1:size(init_rects,1)
    figure()
    subplot(2,1,1);
    plot(x,diffpos(:,i,1))
    
    subplot(2,1,2);
    plot(x,diffpos(:,i,2))
end
end