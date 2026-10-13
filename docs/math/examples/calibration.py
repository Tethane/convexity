"""Family 21: recover volatility from synthetic option prices."""

import numpy as np
from convexity.math.calibration import CalibrationProblem, calibrate
from convexity.math.pricing import forward_option

strikes = np.linspace(80.0, 120.0, 9)


def predict(parameters):
    return np.array([forward_option(100.0, k, parameters[0], 1.0) for k in strikes])


result = calibrate(
    CalibrationProblem(predict, predict([0.24]), [0.4], lower=[0.01], upper=[1.0])
)
print("Parameters:", result.parameters)
print("Identified:", result.identified, "RMSE:", result.rmse)
