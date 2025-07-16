% plot_sjnr_vs_jsr.m
% Sweeps MIMO antenna configs and pre‑filter JSR to plot post‑filter SJNR.

clear; clc; close all;

%% 1) Define antenna configurations and JSR sweep
% Rows are [Nt, Nr]
antenna_configs = [2 2;   % 2×2
                   4 4;   % 4×4
                   8 8];  % 8×8
jsr_dB_range    = -60:10:100;

%% 2) Preallocate result matrix
numConfigs = size(antenna_configs,1);
numJSRs     = numel(jsr_dB_range);
postSJNR_dB = zeros(numConfigs, numJSRs);

%% 3) Loop over configs and JSRs
for i = 1:numConfigs
    Nt = antenna_configs(i,1);
    Nr = antenna_configs(i,2);
    for j = 1:numJSRs
        % Fill sim params
        params.Nt     = Nt;
        params.Nr     = Nr;
        params.JSR_dB = jsr_dB_range(j);
        
        % --- CALL YOUR CORE SIMULATION HERE ---
        % Must return a struct `out` with field `out.postSJNR_dB`
        out = main_simulation(params);  
        % If your function is named `simulate_endtoend`, use that instead:
        % out = simulate_endtoend(params);
        
        % Extract and store post‑filter SJNR
        postSJNR_dB(i,j) = out.postSJNR_dB;
    end
end

%% 4) Plotting
figure('Color','w'); hold on; grid on;
styles = {'-o','-s','-^'};  % one style per config
for i = 1:numConfigs
    plot(jsr_dB_range, postSJNR_dB(i,:), styles{i}, ...
         'LineWidth',1.5, 'MarkerSize',6);
end
xlabel('Pre‑filter JSR (dB)','FontSize',12);
ylabel('Post‑filter SJNR (dB)','FontSize',12);
title('SJNR vs. JSR for Different MIMO Configurations','FontSize',14);
legend('2×2','4×4','8×8','Location','Best','FontSize',10);

% Expand y‑limits for clarity
ymin = floor(min(postSJNR_dB(:))) - 5;
ymax = ceil (max(postSJNR_dB(:))) + 5;
ylim([ymin, ymax]);

%% 5) Save figure
exportgraphics(gcf, 'sjnr_vs_jsr.png', 'Resolution',300);
fprintf('Saved plot as sjnr_vs_jsr.png\n');
