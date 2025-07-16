% plot_constellation_ber.m
% Simulates QPSK constellation at different JSRs and computes inferred BER from pSJNR

clear; clc; close all;

% Create results folder if needed
if ~exist('results', 'dir'); mkdir('results'); end

%% Parameters
Nt = 2; Nr = 2;
numSymbols = 1e4;        % for good BER/SJNR estimate
JSR_dB_list = [-Inf, 0, 10, 20];
SNR_dB = 20;             % fixed

% Gray-coded QPSK mapping (for BER)
symbolMap = [1+1j; -1+1j; -1-1j; 1-1j] / sqrt(2);

for i = 1:length(JSR_dB_list)
    JSR_dB = JSR_dB_list(i);

    %% 1. Generate bits and QPSK symbols
    bits = randi([0 1], numSymbols, 2);
    txSymbolsFlat = qam_modulation(bits).';      % 1 x N
    txSymbols = repmat(txSymbolsFlat, Nt, 1);    % Nt x N

    %% 2. Simulate channel and jamming
    [rxSignal, H] = mimo_channel_simulation(txSymbols, Nt, Nr, SNR_dB, 'none');
    if isfinite(JSR_dB)
        sigPow = mean(abs(rxSignal(:)).^2);
        jamPow = sigPow * 10^(JSR_dB/10);
        jammer = sqrt(jamPow/2) * (randn(size(rxSignal)) + 1j*randn(size(rxSignal)));
    else
        jammer = zeros(size(rxSignal));
    end
    rxWithJam = rxSignal + jammer;

    %% 3. Apply ZF (BJM-style) filter
    P = pinv(H);
    rxFiltered = P * rxWithJam;      % Nt x N
    rxSymbols = mean(rxFiltered, 1); % collapse antennas for scatter

    %% 4. Plot constellation
    figure('Color','w'); scatter(real(rxSymbols), imag(rxSymbols), 10, 'b', 'filled');
    axis square; grid on;
    title(sprintf('QPSK Constellation at JSR = %g dB', JSR_dB));
    xlabel('In-Phase'); ylabel('Quadrature');
    exportgraphics(gcf, sprintf('results/constellation_%gdB.png', JSR_dB), 'Resolution',300);

    %% 5. Compute pSJNR and inferred raw BER
    desired = mean(P * rxSignal, 1); % desired signal (no noise)
    error   = rxSymbols - desired;
    pSJNR   = mean(abs(desired).^2) / mean(abs(error).^2);
    pSJNR_dB = 10 * log10(pSJNR);
    gamma = 10^(pSJNR_dB / 10);
    rawBER = 2*qfunc(sqrt(gamma)) - qfunc(sqrt(gamma))^2;

    %% 6. Log to console
    fprintf('JSR = %5g dB | pSJNR = %6.2f dB | Raw BER ≈ %.2e\n', JSR_dB, pSJNR_dB, rawBER);
end
