% ber_sinr.m
% Computes raw BER and post-SINR based on true bits and received symbols
% Inputs:
%   bits_true: Nx2 binary matrix (original transmitted bits)
%   symbols_rx: 1xN or Nx1 complex symbols (equalized, noisy)
% Outputs:
%   metrics: struct with rawBER and SINR in dB
function metrics = ber_sinr(bits_true, symbols_rx)
    % QPSK decision boundaries (Gray decoding)
    symbols_rx = symbols_rx(:).';
    decisions = zeros(length(symbols_rx), 2);

    for k = 1:length(symbols_rx)
        sym = symbols_rx(k);
        decisions(k,1) = real(sym) < 0;  % 0 → right half, 1 → left half
        decisions(k,2) = imag(sym) < 0;  % 0 → upper half, 1 → lower half
    end

    % Raw BER
    numErrors = sum(bits_true(:) ~= decisions(:));
    totalBits = numel(bits_true);
    rawBER = numErrors / totalBits;

    % Compute SINR
    symbol_est = qam_modulation(decisions);
    errPower = mean(abs(symbols_rx - symbol_est.').^2);
    sigPower = mean(abs(symbol_est).^2);
    SINR = sigPower / errPower;

    metrics.rawBER = rawBER;
    metrics.SINR   = SINR;
    metrics.SINR_dB = 10*log10(SINR);
end

