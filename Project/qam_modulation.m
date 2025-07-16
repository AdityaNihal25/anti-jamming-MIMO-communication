% qam_modulation.m
% Maps input bits to QPSK symbols using Gray coding
% Inputs:
%   bits: N×2 binary matrix (each row = 2 bits)
% Outputs:
%   symbols: N×1 complex QPSK symbols
function symbols = qam_modulation(bits)
    if size(bits,2) ~= 2
        error('QPSK requires exactly 2 bits per symbol.');
    end

    % Gray coded QPSK mapping: 00 → 1+1j, 01 → -1+1j, 11 → -1-1j, 10 → 1-1j
    symbolMap = [1+1j; -1+1j; -1-1j; 1-1j] / sqrt(2);
    indices = bi2de(bits, 'left-msb') + 1;
    symbols = symbolMap(indices);
end
