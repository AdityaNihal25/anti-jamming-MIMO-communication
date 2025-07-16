def simulate_sinr(self, modulation, power_level, nulling, features, jammer_label):
    # Use real MATLAB simulation
    sinr, ber = self.eng.main_simulation(
        float(modulation),
        float(power_level),
        float(nulling),
        float(jammer_label),
        nargout=2
    )
    self.last_ber = ber
    return sinr

