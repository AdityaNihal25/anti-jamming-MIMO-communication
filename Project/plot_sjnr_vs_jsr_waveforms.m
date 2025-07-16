clear; clc; close all;

antenna_configs = [8 8];  % Focus on 8x8 as in paper Fig. 12
jsr_dB_range    = -60:10:100;
jammer_types    = {'broadband', 'tone', 'partial', 'reactive'};

numJSRs = length(jsr_dB_range);
numTypes = length(jammer_types);
postSJNR_dB = zeros(numTypes, numJSRs);

params.Nt = antenna_configs(1);
params.Nr = antenna_configs(2);

for w = 1:numTypes
    params.jammerType = jammer_types{w};
    for j = 1:numJSRs
        params.JSR_dB = jsr_dB_range(j);
        out = main_simulation(params);
        postSJNR_dB(w,j) = out.postSJNR_dB;
    end
end

% Plot all jammer waveforms on same plot
figure; hold on; grid on;
colors = lines(numTypes);
for w = 1:numTypes
    plot(jsr_dB_range, postSJNR_dB(w,:), 'Color', colors(w,:), 'LineWidth',1.5);
end
xlabel('Pre-filter JSR (dB)');
ylabel('Post-filter SJNR (dB)');
title('Effect of Jammer Waveforms on SJNR (8×8 MIMO)');
legend(jammer_types, 'Location', 'Best');
