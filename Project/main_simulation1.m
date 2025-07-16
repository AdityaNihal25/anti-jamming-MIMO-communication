function out = main_simulation(params)
    Nt     = params.Nt;
    Nr     = params.Nr;
    JSR_dB = params.JSR_dB;

    % Generate QPSK symbols
    numSymbols = 1e4;
    bits = randi([0 1], numSymbols, 2);
    txSymbolsFlat = qam_modulation(bits).';     % Make row vector
    txSymbols = repmat(txSymbolsFlat, Nt, 1);   % Nt×numSymbols


    % Simulate channel + jamming
    [rxSignal, H] = mimo_channel_simulation(txSymbols, Nt, Nr, 20, 'broadband');

    % Add jammer using JSR
    sigPow = mean(abs(rxSignal(:)).^2);
    jamPow = sigPow * 10^(JSR_dB / 10);
    jammer = sqrt(jamPow/2) * (randn(size(rxSignal)) + 1j*randn(size(rxSignal)));
    rxWithJam = rxSignal + jammer;

    % Apply simple BJM filter (ZF)
    P = pinv(H);
    rxFiltered = P * rxWithJam;
    s_des = P * rxSignal;
    err = rxFiltered - s_des;
    SJNR = mean(abs(s_des(:)).^2) / mean(abs(err(:)).^2);
    out.postSJNR_dB = 10*log10(SJNR);
end
