function plot_brain(file_path, T, plots_path, method_name)
    if nargin < 1
        file_path = "../tadpole_prediction/tadpole_extrap_50dim_test/000_00.mat";
        T = 3;
        plots_path = "plots/";
        method_name = "best";
    end
    
    load(file_path, 'data'); % --> data
    switch method_name
        case 'real'
            x = data{1};
        case 'best'
            x = data{3};
        case 'mean'
            x = squeeze(mean(data{2}));
        case 'extrapolation'
            x = data{6};
    end

    mkdir(plots_path)
    max_x = 1.7749; %2.3348;
    min_x = 1.0580; %0.6141;
    x = min(x, max_x);
    x = max(x, min_x);
    x = (x - min_x)./(max_x - min_x);
    x = max(x, 0); x = min(x, 1);


    for i=1:size(x, 1)
        fprintf("Sample %d/%d", i, size(x, 1))
        sample = transpose(reshape(x(i, :, :), size(x, 2), size(x, 3)));
        % 1.7749    1.6203    1.7726 - real max
        % 1.1106    1.0580    1.1145 - real min
%         min_x = min(sample(:)); %normalization by all axes
%         max_x = max(sample(:));
%         sample = (sample - min_x) / (max_x - min_x);
        
        % sample should be of size (82, 3) = (brain_dimension, time)
        % Go through all samples and generate separate plots
        plot_brain_(sample, T, plots_path + i, method_name)
    end
end


function plot_brain_(x, T, plots_path, method_name)
    % file_path - load data from
    % out_dir - directory to save files
    % T - time step to do prediction
    % method_name - name of the method to predict
    %clear all;

    % export_fig is a matlab toolbox for saving high quality figures
    %addpath('./altmany-export_fig-76bd7fa/');
    addpath('./export_fig/');
    addpath('./NIfTI_20140122/');

    % N x T matrix of longitudinal outputs for N ROIs (Desikan) and T time
    % points. In my case, av45_adas_gen.mat has av45_gen which is 82x3 matrix.
    % Load your own values here.
    % load av45_adas_gen av45_gen;
    % My change - debug START
    %load '../tadpole_code_extract_data/processed_data/adas13/x.mat' 
    %x = transpose(reshape(X(202,:,:), 3, 82));
    % My change - debug END

    % The ROI figures are from the IIT Desikan website. TADPOLE also uses IIT
    % Desikan atlas but has (1) 82 ROIs and (2) in a different order. Use this
    % mapping such that the i'th ROI in IIT is iit_to_tadpole_idx(i)'s index in
    % TADPOLE.
    load iit_to_tadpole_idx.mat;

    % Sometimes it's better to rescale the values across the ROIs and time
    % points for more apparent changes across time. Try different ways of
    % normalization.
%     min_x = min(x(:)); %normalization by all axes
%     max_x = max(x(:));
%     x = (x - min_x) / (max_x - min_x);
%     
%     min_x = min(x); %normalization for each time axis
%     max_x = max(x);
%     x = (x - min_x) ./ (max_x - min_x);


    %% Color-coding the ROIs
    close all;

    % glass brain
    load glass_fv;
    hold on;

    axis image;
    view([0 0 1]);
    shading interp

    % Desikan ROI glasses
    nii = load_nii('IIT_GM_Desikan_atlas.nii.gz');
    PIB_ROI = double(nii.img);
    cm = colormap('jet');  % Try other colormap if you want
    roi16_idx = unique(PIB_ROI);

    for i=1:length(roi16_idx)
        % Each voxel of PIB_ROI has the ROI index as its value
        v = (PIB_ROI == roi16_idx(i));
%         v = v(1:2, :, :);

        % Turn this on to view the inside of the right hemisphere only (idx < 42)
    %     if iit_to_tadpole_idx(i) ~= -1 && iit_to_tadpole_idx(i) < 42
        if iit_to_tadpole_idx(i) ~= -1
            v = smooth3(v, 'gaussian', 11, 2);      % slight smoothing of the surface is useful
            fv = isosurface(v);                     % Turn the voxels into vertices and triangular faces
            glass_pib.v = v;
            glass_pib.fv = fv;

            % Color-corading
            hold on;
            p = patch(glass_pib.fv);
            % iit_to_tadpole_idx is used here to retrieve the value from the
            % correct ROI in TADPOLE. Scale it and round it for colormap.
            p.FaceColor = cm(round(x(iit_to_tadpole_idx(i), T)*(size(cm, 1) - 1))+1, :);
            % Some other visualization properties that I found to be decent.
            p.EdgeColor = 'none';
            p.SpecularStrength = 0.4;
            p.DiffuseStrength = 0.6;
            p.AmbientStrength = 0.3;
            hold off;
        end
    end
    % Camera lights that I found to be useful.
    camlight(45, 45);
    camlight(-45, -45);
    axis off;


    %% Export figs with transparent background and reasonable dpi in Top, Front, and side view.
    % I highly suggest using export_fig
    %view([-90 90])
    %mkdir(plots_path)
    export_fig([plots_path + "_top_t" + T + "_" + method_name + ".png"], '-transparent', '-r100');

    set(gca,'position',[0 0 1 1], 'units', 'normalized')
    view([1 0 0])
    export_fig([plots_path + "_front_t" + T + "_" + method_name + ".png"], '-transparent', '-r100');

    set(gca,'position',[0 0 1 1], 'units', 'normalized')
    view([0 1 0])
    export_fig([plots_path + "_left_t" + T + "_" + method_name + ".png"], '-transparent', '-r100');

    set(gca,'position',[0 0 1 1], 'units', 'normalized')
    view([0 -1 0])
    export_fig([plots_path + "_right_t" + T + "_" + method_name + ".png"], '-transparent', '-r100');
    
%     ax = axes;
%     c = colorbar(ax, 'fontsize', 16);
%     c.Location = 'southoutside';
%     c.LineWidth = 1
%     ax.Visible = 'off';
%     export_fig([plots_path + "tadpole_cbar.png"], '-transparent', '-r100');

end
