import matlab.engine
eng = matlab.engine.start_matlab()
sinr, ber = eng.main_simulation(0.0, 1.0, 2.0, 1.0, nargout=2)
print("SINR:", sinr, "BER:", ber)