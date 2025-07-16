% sync_test.m
% Test synchronization robustness by plotting cross-correlation of received preamble
% with and without BJM filtering, under various jammer powers

clear; clc; close all;

%% Parameters
Nt = 2; Nr = 2;
numSymbols = 512;           % Length of preamble
SNR_dB = 20;
JSR_dB_list = [0, 10, 20];   % Jammer powers to test

%% Generate known QPSK preamble
bits = randi([0 1], numSymbols, 2);
txPreamble = qam_modulation(bits).';   % 1 x N
preambleTx = repmat(txPreamble, Nt, 1);  % Nt x N

%% Simulate channel
[rxSignal, H] = mimo_channel_simulation(preambleTx, Nt, Nr, SNR_dB, 'none');

%% Loop through JSR values
for k = 1:length(JSR_dB_list)
    JSR_dB = JSR_dB_list(k);

    % Generate jammer signal
    sigPow = mean(abs(rxSignal(:)).^2);
    jamPow = sigPow * 10^(JSR_dB/10);
    jammer = sqrt(jamPow/2) * (randn(size(rxSignal)) + 1j*randn(size(rxSignal)));

    % Add jammer
    rxWithJam = rxSignal + jammer;

    %% 1. No filtering
    combined_noFilter = sum(rxWithJam, 1);
    [xc_noFilter, lag] = xcorr(combined_noFilter, txPreamble);

    %% 2. Apply ZF filter (BJM-style)
    P = pinv(H);
    rxFiltered = P * rxWithJam;
    combined_bjm = sum(rxFiltered, 1);
    [xc_bjm, ~] = xcorr(combined_bjm, txPreamble);

    %% Plot
    figure('Color','w'); hold on;
    plot(lag, abs(xc_noFilter), 'r--', 'LineWidth',1.5);
    plot(lag, abs(xc_bjm), 'b-', 'LineWidth',1.5);
    xlabel('Lag'); ylabel('Correlation');
    title(sprintf('Preamble Correlation at JSR = %d dB', JSR_dB));
    legend('No Filter','BJM Filter','Location','Best');
    grid on;
    exportgraphics(gcf, sprintf('sync_peak_%ddB.png', JSR_dB), 'Resolution', 300);
end
