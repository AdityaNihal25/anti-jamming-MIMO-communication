% mimo_channel_simulation.m
% Simulate an Nt×Nr MIMO channel with AWGN noise and optional jamming.
% Usage:
%   [rxSignal, H] = mimo_channel_simulation(txSymbols, Nt, Nr, SNR_dB, jammerType)
% Inputs:
%   txSymbols : Nt x numSymbols matrix of transmitted symbols
%   Nt        : number of transmit antennas
%   Nr        : number of receive antennas
%   SNR_dB    : (optional) Signal-to-noise ratio in dB (default: 20 dB)
%   jammerType: (optional) 'none','broadband','tone','partial','reactive' (default: 'none')
% Outputs:
%   rxSignal  : Nr x numSymbols matrix of received symbols after channel, noise, jammer
%   H         : Nr x Nt MIMO channel matrix

function [rxSignal, H] = mimo_channel_simulation(txSymbols, Nt, Nr, SNR_dB, jammerType)
    % Validate inputs
    if nargin < 4 || isempty(SNR_dB)
        SNR_dB = 20; % default SNR
    end
    if nargin < 5 || isempty(jammerType)
        jammerType = 'none';
    end

    % Determine number of symbols
    [nTx, numSymbols] = size(txSymbols);
    if nTx ~= Nt
        error('txSymbols must be dimension Nt x numSymbols');
    end

    % Generate flat-fading Rayleigh channel
    H = (randn(Nr, Nt) + 1j*randn(Nr, Nt)) / sqrt(2);

    % Pass symbols through channel
    rxClean = H * txSymbols;

    % Compute signal power for each Rx antenna
    sigPow = mean(abs(rxClean(:)).^2);

    % AWGN noise generation
    SNR_lin   = 10^(SNR_dB/10);
    noisePow  = sigPow / SNR_lin;
    noise = sqrt(noisePow/2) * (randn(size(rxClean)) + 1j*randn(size(rxClean)));

    % Jammer generation
    switch lower(jammerType)
        case 'broadband'
            jamPow = sigPow; % equal power
            jammer = sqrt(jamPow/2) * (randn(size(rxClean)) + 1j*randn(size(rxClean)));
        case 'tone'
            jamPow = sigPow;
            t = 0:(numSymbols-1);
            tone = exp(1j*2*pi*0.1*t); % tone frequency normalized
            jammer = repmat(tone, Nr, 1) * sqrt(jamPow);
        case 'partial'
            jamPow = sigPow;
            mask = rand(Nr, numSymbols) > 0.5;
            jammer = sqrt(jamPow/2) * ((randn(Nr, numSymbols) + 1j*randn(Nr, numSymbols)) .* mask);
        case 'reactive'
            % Example: jammer acts when instantaneous RX amplitude crosses a threshold
            jamPow = sigPow;
            mask = abs(rxClean) > sqrt(sigPow);
            jammer = sqrt(jamPow/2) * (randn(size(rxClean)) + 1j*randn(size(rxClean))) .* mask;
        case 'none'
            jammer = zeros(size(rxClean));
        otherwise
            error('Unsupported jammer type: %s', jammerType);
    end

    % Total received signal
    rxSignal = rxClean + noise + jammer;
end
