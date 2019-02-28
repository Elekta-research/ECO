function [runparams] = step(params, seq, im, runparams)
tic();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Target localization step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Do not estimate translation and scaling on the first frame, since we 
% just want to initialize the tracker there
if seq.frame > 1
    old_pos = inf(size(runparams.pos));
    iter = 1;
    
    %translation search
    while iter <= params.refinement_iterations && any(old_pos ~= runparams.pos)
        % Extract features at multiple resolutions
        sample_pos = round(runparams.pos);
        runparams.det_sample_pos = sample_pos;
        sample_scale = runparams.currentScaleFactor*params.scaleFactors;
        xt = extract_features(im, sample_pos, sample_scale, params.features, params.global_fparams, params.feature_extract_info);
    
        % Project sample
        xt_proj = project_sample(xt, runparams.projection_matrix);
    
        % Do windowing of features
        cos_window = params.cos_window;
        xt_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj, cos_window, 'uniformoutput', false);
        params.cos_window = cos_window;
        % Compute the fourier series
        xtf_proj = cellfun(@cfft2, xt_proj, 'uniformoutput', false);
    
        % Interpolate features to the continuous domain
        xtf_proj = interpolate_dft(xtf_proj, params.interp1_fs, params.interp2_fs);
    
        % Compute convolution for each feature block in the Fourier domain
        % and the sum over all blocks.
        runparams.scores_fs_feat{params.k1} = sum(bsxfun(@times, runparams.hf_full{params.k1}, xtf_proj{params.k1}), 3);
        scores_fs_sum = runparams.scores_fs_feat{params.k1};
        for k = params.block_inds
            runparams.scores_fs_feat{k} = sum(bsxfun(@times, runparams.hf_full{k}, xtf_proj{k}), 3);
            scores_fs_sum(1+params.pad_sz{k}(1):end-params.pad_sz{k}(1), 1+params.pad_sz{k}(2):end-params.pad_sz{k}(2),1,:) = ...
                scores_fs_sum(1+params.pad_sz{k}(1):end-params.pad_sz{k}(1), 1+params.pad_sz{k}(2):end-params.pad_sz{k}(2),1,:) + ...
                runparams.scores_fs_feat{k};
        end
        % Also sum over all feature blocks.
        % Gives the fourier coefficients of the convolution response.
        runparams.scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
       
        % Optimize the continuous score function with Newton's method.
        [trans_row, trans_col, runparams.scale_ind] = optimize_scores(runparams.scores_fs, params.newton_iterations);
        % Compute the translation vector in pixel-coordinates and round
        % to the closest integer pixel.
        translation_vec = [trans_row, trans_col] .* (params.img_support_sz./params.output_sz) * runparams.currentScaleFactor * params.scaleFactors(runparams.scale_ind);
        scale_change_factor = params.scaleFactors(runparams.scale_ind);
        
        % update position
        old_pos = runparams.pos;
        runparams.pos = sample_pos + translation_vec;
        
        if params.clamp_position
            runparams.pos = max([1 1], min([size(im,1) size(im,2)], runparams.pos));
        end
        % Do scale tracking with the scale filter
        if params.nScales > 0 && params.use_scale_filter
            scale_change_factor = scale_filter_track(im, runparams.pos, params.base_target_sz, runparams.currentScaleFactor, runparams.scale_filter, params);
        end
    
        % Update the scale
        runparams.currentScaleFactor = runparams.currentScaleFactor * scale_change_factor;
        % Adjust to make sure we are not to large or to small
        if runparams.currentScaleFactor < params.min_scale_factor
            runparams.currentScaleFactor = params.min_scale_factor;
        elseif runparams.currentScaleFactor > params.max_scale_factor
            runparams.currentScaleFactor = params.max_scale_factor;
        end
        iter = iter + 1;
    end
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model update step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract sample and init projection matrix
if seq.frame == 1
    % Extract image region for training sample
    sample_pos = round(runparams.pos);
    sample_scale = runparams.currentScaleFactor;
    xl = extract_features(im, sample_pos, runparams.currentScaleFactor, params.features, params.global_fparams, params.feature_extract_info);
    
    % Do windowing of features
    cos_window = params.cos_window;
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
        
    % Compute the fourier series
    xlf = cellfun(@cfft2, xlw, 'uniformoutput', false);
       
    % Interpolate features to the continuous domain
    xlf = interpolate_dft(xlf, params.interp1_fs, params.interp2_fs);
        
    % New sample to be added
    xlf = compact_fourier_coeff(xlf);
    % Shift sample
    shift_samp = 2*pi * (runparams.pos - sample_pos) ./ (sample_scale * params.img_support_sz);
    xlf = shift_sample(xlf, shift_samp, params.kx, params.ky);
    
    % Init the projection matrix
    runparams.projection_matrix = init_projection_matrix(xl, params.sample_dim, params);
        
    % Project sample
    xlf_proj = project_sample(xlf, runparams.projection_matrix);        
    clear xlw
elseif params.learning_rate > 0
    if ~params.use_detection_sample
        % Extract image region for training sample
        sample_pos = round(runparams.pos);
        sample_scale = runparams.currentScaleFactor;
        xl = extract_features(im, sample_pos, runparams.currentScaleFactor, params.features, params.global_fparams, params.feature_extract_info);
            
        % Project sample
        xl_proj = project_sample(xl, runparams.projection_matrix);
            
        % Do windowing of features
        xl_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, params.cos_window), xl_proj, params.cos_window, 'uniformoutput', false);
            
        % Compute the fourier series
        xlf1_proj = cellfun(@cfft2, xl_proj, 'uniformoutput', false);
        % Interpolate features to the continuous domain
        xlf1_proj = interpolate_dft(xlf1_proj, params.interp1_fs, params.interp2_fs);
            
        % New sample to be added
        xlf_proj = compact_fourier_coeff(xlf1_proj);
    else
        if params.debug
            % Only for visualization
            xl = cellfun(@(xt) xt(:,:,:,runparams.scale_ind), xt, 'uniformoutput', false);
        end
        % Use the sample that was used for detection
        sample_scale = sample_scale(runparams.scale_ind);
        xlf_proj = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:,runparams.scale_ind), xtf_proj, 'uniformoutput', false);
    end
    % Shift the sample so that the target is centered
    shift_samp = 2*pi * (runparams.pos - sample_pos) ./ (sample_scale * params.img_support_sz);
    xlf_proj = shift_sample(xlf_proj, shift_samp, params.kx, params.ky);
end

% The permuted sample is only needed for the CPU implementation
if ~params.use_gpu
    xlf_proj_perm = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj, 'uniformoutput', false);
end

if params.use_sample_merge
    % Update the samplesf to include the new sample. The distance
    % matrix, kernel matrix and prior weight are also updated
    if params.use_gpu
        [merged_sample, new_sample, merged_sample_id, new_sample_id, runparams.distance_matrix, runparams.gram_matrix, runparams.prior_weights] = ...
            update_sample_space_model_gpu(runparams.samplesf, xlf_proj, runparams.distance_matrix, runparams.gram_matrix, runparams.prior_weights,...
            runparams.num_training_samples,params);
    else
        [merged_sample, new_sample, merged_sample_id, new_sample_id, runparams.distance_matrix, runparams.gram_matrix, runparams.prior_weights] = ...
            update_sample_space_model(runparams.samplesf, xlf_proj_perm, runparams.distance_matrix, runparams.gram_matrix, runparams.prior_weights,...
            runparams.num_training_samples,params);
    end
    if runparams.num_training_samples < params.nSamples
        runparams.num_training_samples = runparams.num_training_samples + 1;
    end
else
    % Do the traditional adding of a training sample and weight update
    % of C-COT
    [runparams.prior_weights, replace_ind] = update_prior_weights(runparams.prior_weights, gather(runparams.sample_weights), runparams.latest_ind, seq.frame, params);
    runparams.latest_ind = replace_ind;
    merged_sample_id = 0;
    new_sample_id = replace_ind;
    if params.use_gpu
        new_sample = xlf_proj;
    else
        new_sample = xlf_proj_perm;
    end
end

if seq.frame > 1 && params.learning_rate > 0 || seq.frame == 1 && ~params.update_projection_matrix
    % Insert the new training sample
    for k = 1:runparams.num_feature_blocks
        if params.use_gpu
            if merged_sample_id > 0
                runparams.samplesf{k}(:,:,:,merged_sample_id) = merged_sample{k};
            end
            if new_sample_id > 0
                runparams.samplesf{k}(:,:,:,new_sample_id) = new_sample{k};
            end
        else
            if merged_sample_id > 0
                runparams.samplesf{k}(merged_sample_id,:,:,:) = merged_sample{k};
            end
            if new_sample_id > 0
                runparams.samplesf{k}(new_sample_id,:,:,:) = new_sample{k};
            end
        end
    end
end

runparams.sample_weights = cast(runparams.prior_weights, 'like', params.data_type);
train_tracker = (seq.frame < params.skip_after_frame) || (runparams.frames_since_last_train >= params.train_gap);
    
if train_tracker     
    % Used for preconditioning
    new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf_proj, 'uniformoutput', false);
        
    if seq.frame == 1
        % Initialize stuff for the filter learning
            
        % Initialize Conjugate Gradient parameters
        runparams.sample_energy = new_sample_energy;
        runparams.CG_state = [];
            
        if params.update_projection_matrix
            % Number of CG iterations per GN iteration
            runparams.init_CG_opts.maxit = ceil(params.init_CG_iter / params.init_GN_iter);
            
            runparams.hf = cell(2,1,runparams.num_feature_blocks);
            yf = params.yf;
            proj_energy = cellfun(@(P, yf) 2*sum(abs(yf(:)).^2) / sum(params.feature_dim) * ones(size(P), 'like', params.data_type), runparams.projection_matrix, yf, 'uniformoutput', false);
        else
            runparams.CG_opts.maxit = params.init_CG_iter; % Number of initial iterations if projection matrix is not updated
            
            runparams.hf = cell(1,1,runparams.num_feature_blocks);
        end
        % Initialize the filter with zeros
        for k = 1:runparams.num_feature_blocks
            runparams.hf{1,1,k} = zeros([params.filter_sz(k,1) (params.filter_sz(k,2)+1)/2 params.sample_dim(k)], 'like', params.data_type_complex);
        end
    else
        runparams.CG_opts.maxit = params.CG_iter;
            
        % Update the approximate average sample energy using the learning
        % rate. This is only used to construct the preconditioner.
        runparams.sample_energy = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, runparams.sample_energy, new_sample_energy, 'uniformoutput', false);
    end
    
    % Do training
    if seq.frame == 1 && params.update_projection_matrix
        if params.debug
            projection_matrix_init = runparams.projection_matrix;
        end
        % Initial Gauss-Newton optimization of the filter and
        % projection matrix.
        if params.use_gpu
            [runparams.hf, runparams.projection_matrix, runparams.res_norms] = train_joint_gpu(runparams.hf, runparams.projection_matrix, xlf, params.yf, params.reg_filter, runparams.sample_energy, params.reg_energy, proj_energy, params, runparams.init_CG_opts);
        else
            [runparams.hf, runparams.projection_matrix, runparams.res_norms] = train_joint(runparams.hf, runparams.projection_matrix, xlf, params.yf, params.reg_filter, runparams.sample_energy, params.reg_energy, proj_energy, params, runparams.init_CG_opts);
        end
        
        % Re-project and insert training sample
        xlf_proj = project_sample(xlf, runparams.projection_matrix);
        for k = 1:runparams.num_feature_blocks
            if params.use_gpu
                runparams.samplesf{k}(:,:,:,1) = xlf_proj{k};
            else
                runparams.samplesf{k}(1,:,:,:) = permute(xlf_proj{k}, [4 3 1 2]);
            end
        end
        % Update the gram matrix since the sample has changed
        if strcmp(params.distance_matrix_update_type, 'exact')
            % Find the norm of the reprojected sample
            new_train_sample_norm =  0;
                
            for k = 1:runparams.num_feature_blocks
                new_train_sample_norm = new_train_sample_norm + real(gather(2*(xlf_proj{k}(:)' * xlf_proj{k}(:))));% - reshape(xlf_proj{k}(:,end,:,:), [], 1, 1)' * reshape(xlf_proj{k}(:,end,:,:), [], 1, 1));
            end
            runparams.gram_matrix(1,1) = new_train_sample_norm;
        end
        if params.debug
            norm_proj_mat_init = sqrt(sum(cellfun(@(P) gather(norm(P(:))^2), projection_matrix_init)));
            norm_proj_mat = sqrt(sum(cellfun(@(P) gather(norm(P(:))^2), runparams.projection_matrix)));
            norm_proj_mat_change = sqrt(sum(cellfun(@(P,P2) gather(norm(P(:) - P2(:))^2), projection_matrix_init, runparams.projection_matrix)));
            fprintf('Norm init: %f, Norm final: %f, Matrix change: %f\n', norm_proj_mat_init, norm_proj_mat, norm_proj_mat_change / norm_proj_mat_init);
        end
    else
        % Do Conjugate gradient optimization of the filter
        if params.use_gpu
            [runparams.hf, runparams.res_norms, runparams.CG_state] = train_filter_gpu(runparams.hf, runparams.samplesf, params.yf, params.reg_filter, runparams.sample_weights, runparams.sample_energy, params.reg_energy, params, runparams.CG_opts, runparams.CG_state);
        else
            [runparams.hf, runparams.res_norms, runparams.CG_state] = train_filter(runparams.hf, runparams.samplesf, params.yf, params.reg_filter, runparams.sample_weights, runparams.sample_energy, params.reg_energy, params, runparams.CG_opts, runparams.CG_state);
        end
    end
    
    % Reconstruct the full Fourier series
    runparams.hf_full = full_fourier_coeff(runparams.hf);
        
    runparams.frames_since_last_train = 0;
else
   runparams.frames_since_last_train = runparams.frames_since_last_train+1;
end

% Update the scale filter
if params.nScales > 0 && params.use_scale_filter
    runparams.scale_filter = scale_filter_update(im, runparams.pos, params.base_target_sz, runparams.currentScaleFactor, params.scale_filter, params);
end

% Update the target size (only used for computing output box)
runparams.target_sz = params.base_target_sz * runparams.currentScaleFactor;
    
%save position and calculate FPS
tracking_result.center_pos = double(runparams.pos);
tracking_result.target_size = double(runparams.target_sz);
seq = report_tracking_result(seq, tracking_result);
    
seq.time = seq.time + toc();
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% debug visualization
if params.debug
    figure(20)
    %         set(gcf,'units','normalized','outerposition',[0 0 1 1]);
    subplot_cols = runparams.num_feature_blocks;
    subplot_rows = 3;%ceil(feature_dim/subplot_cols);
    for disp_layer = 1:runparams.num_feature_blocks
        subplot(subplot_rows,subplot_cols,disp_layer);
        imagesc(mean(abs(sample_fs(conj(runparams.hf_full{disp_layer}))), 3));
        colorbar;
        axis image;
        subplot(subplot_rows,subplot_cols,disp_layer+subplot_cols);
        imagesc(mean(abs(xl{disp_layer}), 3));
        colorbar;
        axis image;
        if seq.frame > 1
            subplot(subplot_rows,subplot_cols,disp_layer+2*subplot_cols);
            imagesc(fftshift(sample_fs(runparams.scores_fs_feat{disp_layer}(:,:,1,runparams.scale_ind))));
            colorbar;
            axis image;
        end
    end
    
    if train_tracker
        runparams.residuals_pcg = [runparams.residuals_pcg; runparams.res_norms];
        res_start_ind = max(1, length(runparams.residuals_pcg)-300);
        figure(99);plot(res_start_ind:length(runparams.residuals_pcg), runparams.residuals_pcg(res_start_ind:end));
        axis([res_start_ind, length(runparams.residuals_pcg), 0, min(max(runparams.residuals_pcg(res_start_ind:end)), 0.2)]);
    end
end



% close(writer);

[seq, runparams.results] = get_sequence_results(seq);

%disp(['fps: ' num2str(runparams.results.fps)])
end

