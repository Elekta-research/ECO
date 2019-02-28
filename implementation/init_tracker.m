function [runparams, params] = init_tracker(params, seq, im, is_color_image)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info
%[runparams.seq, im] = get_sequence_info(params.seq);
%params = rmfield(params, 'seq');
%if isempty(im)
%    runparams.seq.rect_position = [];
%    [runparams.seq, runparams.results] = get_sequence_results(runparams.seq);
%    return;
%end

% Init position
runparams.pos = seq.init_pos(:)';
%poss = seq.init_poss(:)';
target_sz = seq.init_sz(:)';
%targets_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
params.features = params.t_features; % TODO

% Set default seq
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Correct max number of samples
params.nSamples = min(params.nSamples, seq.num_frames);

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    warning('ECO:tracker', 'Error when using the mexResize function. Using Matlab''s interpolation function instead, which is slower.\nTry to run the compile script in "external_libs/mexResize/".\n\nThe error was:\n%s', getReport(err));
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    runparams.currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    runparams.currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    runparams.currentScaleFactor = 1.0;
end

% target size at the initial scale
params.base_target_sz = target_sz / runparams.currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( params.base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(params.base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = params.base_target_sz + sqrt(prod(params.base_target_sz * params.search_area_scale) + (params.base_target_sz(1) - params.base_target_sz(2))/4) - sum(params.base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [params.base_target_sz(1)*2 params.base_target_sz(2)*2]; % for testing
end

[params.features, params.global_fparams, feature_info] = init_features(params.features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');

% Set feature info
params.img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
params.feature_dim = feature_info.dim;
runparams.num_feature_blocks = length(params.feature_dim);

% Get feature specific parameters
feature_params = init_feature_params(params.features, feature_info);
params.feature_extract_info = get_feature_extract_info(params.features);

% Set the sample feature dimension
if params.use_projection_matrix
    params.sample_dim = feature_params.compressed_dim;
else
    params.sample_dim = params.feature_dim;
end

% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,runparams.num_feature_blocks), 2), [2 3 1]);

% Number of Fourier coefficients to save for each filter layer. This will
% be an odd number.
params.filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(params.filter_sz, ones(1,runparams.num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size.
[params.output_sz, params.k1] = max(params.filter_sz, [], 1);
params.k1 = params.k1(1);

% Get the remaining block indices
params.block_inds = 1:runparams.num_feature_blocks;
params.block_inds(params.k1) = [];

% How much each feature block has to be padded to the obtain params.output_sz
filter_sz = params.filter_sz;
params.pad_sz = cellfun(@(filter_sz) (params.output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false);
params.filter_sz = filter_sz;

% Compute the Fourier series indices and their transposes
ky = cellfun(@(sz) (-ceil((sz(1) - 1)/2) : floor((sz(1) - 1)/2))', filter_sz_cell, 'uniformoutput', false);
kx = cellfun(@(sz) -ceil((sz(2) - 1)/2) : 0, filter_sz_cell, 'uniformoutput', false);

% construct the Gaussian label function using Poisson formula
sig_y = sqrt(prod(floor(params.base_target_sz))) * params.output_sigma_factor * (params.output_sz ./ params.img_support_sz);
yf_y = cellfun(@(ky) single(sqrt(2*pi) * sig_y(1) / params.output_sz(1) * exp(-2 * (pi * sig_y(1) * ky / params.output_sz(1)).^2)), ky, 'uniformoutput', false);
yf_x = cellfun(@(kx) single(sqrt(2*pi) * sig_y(2) / params.output_sz(2) * exp(-2 * (pi * sig_y(2) * kx / params.output_sz(2)).^2)), kx, 'uniformoutput', false);
params.yf = cellfun(@(yf_y, yf_x) cast(yf_y * yf_x, 'like', params.data_type), yf_y, yf_x, 'uniformoutput', false);
params.kx = kx;
params.ky = ky;

% construct cosine window
cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
params.cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

% Compute Fourier series of interpolation function
[params.interp1_fs, params.interp2_fs] = cellfun(@(sz) get_interp_fourier(sz, params), filter_sz_cell, 'uniformoutput', false);

% Get the reg_window_edge parameter
reg_window_edge = {};
for k = 1:length(params.features)
    if isfield(params.features{k}.fparams, 'reg_window_edge')
        reg_window_edge = cat(3, reg_window_edge, permute(num2cell(params.features{k}.fparams.reg_window_edge(:)), [2 3 1]));
    else
        reg_window_edge = cat(3, reg_window_edge, cell(1, 1, length(params.features{k}.fparams.nDim)));
    end
end

% Construct spatial regularization filter
params.reg_filter = cellfun(@(reg_window_edge) get_reg_filter(params.img_support_sz, params.base_target_sz, params, reg_window_edge), reg_window_edge, 'uniformoutput', false);

% Compute the energy of the filter (used for preconditioner)
reg_filter = params.reg_filter;
params.reg_energy = cellfun(@(reg_filter) real(reg_filter(:)' * reg_filter(:)), reg_filter, 'uniformoutput', false);
params.reg_filter = reg_filter;

if params.use_scale_filter
    [nScales, scale_step, scaleFactors, scale_filter, parameters] = init_scale_filter(params);
    params = parameters;
    params.nScales = nScales;
    params.scaleFactors = scaleFactors;
    params.scale_filter = scale_filter;
    
else
    % Use the translation filter to estimate the scale.
    params.nScales = params.number_of_scales;
    scale_step = params.scale_step;
    scale_exp = (-floor((params.nScales-1)/2):ceil((params.nScales-1)/2));
    params.scaleFactors = scale_step .^ scale_exp;
end

if params.nScales > 0
    %force reasonable scale changes
    params.min_scale_factor = scale_step ^ ceil(log(max(5 ./ params.img_support_sz)) / log(scale_step));
    params.max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ params.base_target_sz)) / log(scale_step));
end

% Set conjugate gradient uptions
runparams.init_CG_opts.CG_use_FR = true;
runparams.init_CG_opts.tol = 1e-6;
runparams.init_CG_opts.CG_standard_alpha = true;
runparams.init_CG_opts.debug = params.debug;
runparams.CG_opts.CG_use_FR = params.CG_use_FR;
runparams.CG_opts.tol = 1e-6;
runparams.CG_opts.CG_standard_alpha = params.CG_standard_alpha;
runparams.CG_opts.debug = params.debug;
if params.CG_forgetting_rate == Inf || params.learning_rate >= 1
    runparams.CG_opts.init_forget_factor = 0;
else
    runparams.CG_opts.init_forget_factor = (1-params.learning_rate)^params.CG_forgetting_rate;
end

% Initialize and allocate
runparams.prior_weights = zeros(params.nSamples,1, 'single');
runparams.sample_weights = cast(runparams.prior_weights, 'like', params.data_type);
runparams.samplesf = cell(1, 1, runparams.num_feature_blocks);
if params.use_gpu
    % In the GPU version, the data is stored in a more normal way since we
    % dont have to use mtimesx.
    for k = 1:runparams.num_feature_blocks
        runparams.samplesf{k} = zeros(filter_sz(k,1),(filter_sz(k,2)+1)/2,params.sample_dim(k),params.nSamples, 'like', params.data_type_complex);
    end
else
    for k = 1:runparams.num_feature_blocks
        runparams.samplesf{k} = zeros(params.nSamples,params.sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
    end
end

% Allocate
runparams.scores_fs_feat = cell(1,1,runparams.num_feature_blocks);

% Distance matrix stores the square of the euclidean distance between each pair of
% samples. Initialise it to inf
runparams.distance_matrix = inf(params.nSamples, 'single');

% Kernel matrix, used to update distance matrix
runparams.gram_matrix = inf(params.nSamples, 'single');

runparams.latest_ind = [];
runparams.frames_since_last_train = inf;
runparams.num_training_samples = 0;

% Find the minimum allowed sample weight. Samples are discarded if their weights become lower 
params.minimum_sample_weight = params.learning_rate*(1-params.learning_rate)^(2*params.nSamples);

runparams.res_norms = [];
runparams.residuals_pcg = [];

runparams.projection_matrix = {};
runparams.sample_energy = {};
runparams.CG_state = [];
runparams.hf = {};
runparams.hf_full = {};
runparams.scale_filter = {};
runparams.target_sz = [];
runparams.results = struct;
runparams.det_sample_pos = [];
runparams.scores_fs = [];
runparams.scale_ind = 0;
end