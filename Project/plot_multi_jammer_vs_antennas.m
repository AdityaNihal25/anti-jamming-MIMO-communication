% plot_multi_jammer_vs_antennas.m
% Simulates SJNR vs. number of Rx antennas for 1–3 jammers (K) at fixed JSR

clear; clc; close all;

% Setup
Nt = 2;                            % Fixed number of Tx antennas
K_list = [1, 2, 3];                % Number of jammers
Nr_list = [2, 3, 4, 6, 8];         % Range of Rx antennas
JSR_dB = 20;                       % Fixed jammer power
SNR_dB = 20;                       % Fixed SNR

% Preallocate
postSJNR_dB = zeros(length(K_list), length(Nr_list));

for k = 1:length(K_list)
    K = K_list(k);
    for n = 1:length(Nr_list)
        Nr = Nr_list(n);

        % Generate combined K-jammer signal
        numSymbols = 1e4;
        bits = randi([0 1], numSymbols, 2);
        txSymbolsFlat = qam_modulation(bits).';
        txSymbols = repmat(txSymbolsFlat, Nt, 1);

        [rxSignal, H] = mimo_channel_simulation(txSymbols, Nt, Nr, SNR_dB, 'none');
        sigPow = mean(abs(rxSignal(:)).^2);

        jammer = zeros(size(rxSignal));
        for kk = 1:K
            jam_k = sqrt((sigPow * 10^(JSR_dB/10)) / (2*K)) * ...
                (randn(size(rxSignal)) + 1j*randn(size(rxSignal)));
            jammer = jammer + jam_k;
        end
        rxWithJam = rxSignal + jammer;

        % Apply BJM-style ZF filter
        P = pinv(H);
        rxFiltered = P * rxWithJam;
        s_des = P * rxSignal;
        err = rxFiltered - s_des;
        SJNR = mean(abs(s_des(:)).^2) / mean(abs(err(:)).^2);
        postSJNR_dB(k,n) = 10*log10(SJNR);
    end
end

% Plot
figure('Color','w'); hold on; grid on;
markers = {'-o','-s','-^'};
for k = 1:length(K_list)
    plot(Nr_list, postSJNR_dB(k,:), markers{k}, 'LineWidth',1.5);
end
xlabel('Number of Rx Antennas');
ylabel('Post-filter SJNR (dB)');
title(sprintf('SJNR vs. Rx Antennas for K = 1–3 Jammers @ JSR = %d dB', JSR_dB));
legend('K = 1','K = 2','K = 3','Location','Best');
exportgraphics(gcf, 'results/sjnr_multi_jammer_vs_antennas.png', 'Resolution', 300);
