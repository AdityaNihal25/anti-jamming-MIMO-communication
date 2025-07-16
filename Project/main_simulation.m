% main_simulation.m
% Interface function to be called from Python PPO environment
% Inputs: modulation, power_level, nulling, jammer_label (as scalars)
% Outputs: sinr_out, ber_out (scalars)
function [sinr_out, ber_out] = main_simulation(modulation, power_level, nulling, jammer_label)
% Decode configuration
modulation_schemes = {'qpsk', '16qam', '64qam'};
powers = [1, 2, 3];            % example transmit power levels
nulling_modes = {'none', 'partial', 'full'};

mod_scheme = modulation_schemes{modulation+1};
tx_power = powers(power_level+1);
null_mode = nulling_modes{nulling+1};

Nt = 4; Nr = 4;
numSymbols = 1000;
snr_dB = 20;

% Generate random bits
M = [4, 16, 64];
mod_order = M(modulation+1);
bits_per_symbol = log2(mod_order);
bits = randi([0 1], numSymbols, bits_per_symbol);

% Modulate
symbols = qammod(bi2de(bits), mod_order, 'gray');
txSymbols = repmat(symbols.', Nt, 1);

% Channel simulation
H = (randn(Nr,Nt) + 1i*randn(Nr,Nt))/sqrt(2);
noise = (randn(Nr,numSymbols) + 1i*randn(Nr,numSymbols))/sqrt(2);

% Apply jammer
if jammer_label == 1
    jammer = sqrt(tx_power) * (randn(Nr,numSymbols)+1i*randn(Nr,numSymbols))/sqrt(2);
else
    jammer = zeros(Nr, numSymbols);
end

% RX signal
rx = H * txSymbols + noise + jammer;

% Nulling filter (simplified)
if strcmp(null_mode, 'full')
    rx_filtered = pinv(H) * rx;
elseif strcmp(null_mode, 'partial')
    rx_filtered = H' * rx;
else
    rx_filtered = rx;
end

% Estimate SINR
signal_power = mean(abs(H * txSymbols).^2, 'all');
noise_power = mean(abs(noise).^2, 'all') + mean(abs(jammer).^2, 'all');
sinr_out = 10*log10(signal_power / noise_power);

% Demodulate and BER
rx_flat = mean(rx_filtered,1);
rx_sym = qamdemod(rx_flat, mod_order, 'gray');
rx_bits = de2bi(rx_sym);

% Ensure sizes match for BER computation
min_len = min(size(bits,1), size(rx_bits,1));
bits = bits(1:min_len, :);
rx_bits = rx_bits(1:min_len, :);
bit_errors = sum(bits ~= rx_bits, 'all');
ber_out = bit_errors / numel(bits);

end
