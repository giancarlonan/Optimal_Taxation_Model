# Optimal Taxation Model for Addressing Income Inequality

## Overview

This project models optimal progressive taxation to address rising income inequality, based on the principles and findings from my dissertation. Using a calibrated model of the U.S. economy in 2022, the analysis explores how progressive taxation can maximize social welfare while balancing the trade-offs between equity and efficiency.

The repository contains MATLAB scripts that simulate individual behavioral responses to taxation, compute optimal taxation policies, and perform counterfactual analyses to assess how governments should respond to rising inequality.

---

## Key Features
- **Simulated Wage Distribution**: Generates wage data that reflects income inequality in the U.S., calibrated to match the 2022 Gini coefficient (0.395).
- **Optimal Taxation**: Solves for optimal taxation parameters (`gamma` and `lambda`) to maximize social welfare, considering behavioral responses to taxes and transfers.
- **Counterfactual Analysis**: Explores the effects of increasing income inequality and provides recommendations for adapting taxation policies to sustain economic stability and fairness.
- **Visualization**: Generates plots for Lorenz curves, labor and consumption responses, Gini coefficients, and aggregate macroeconomic metrics.

---

## Files and Functions

### Main Script
- **`Mod_Main.m`**: The entry point of the project. Executes the analysis pipeline:
  - Simulates wage data and computes pre- and post-tax metrics.
  - Solves the optimal taxation problem.
  - Performs counterfactual analysis of rising income inequality.
  - Visualizes results through comprehensive graphs.

### Supporting Functions
1. **`labor.m`**: Computes optimal labor supply for individuals given their wages and taxation parameters.
2. **`cons.m`**: Calculates optimal consumption for individuals based on labor supply, tax progressivity, and government transfers.
3. **`SWF.m`**: Evaluates the social welfare function as the weighted sum of individual utilities.
4. **`transfers.m`**: Determines government transfers based on aggregated tax revenue from individuals.

---

## Usage

1. **Set Up**:
   - Clone the repository or download the files.
   - Ensure all MATLAB files are in the same directory.

2. **Run**:
   - Open MATLAB R2024b (or later) and set the working directory to the folder containing the files.
   - Open and run `Mod_Main.m` to execute the entire analysis.

3. **Outputs**:
   - Graphs illustrating:
     - Lorenz curves and Gini coefficients (pre- and post-taxation).
     - Labor and consumption responses to tax progressivity.
     - Optimal progressivity and transfers under varying inequality.
   - Numerical results for optimal taxation parameters and aggregate metrics.

---

## Results

### Baseline Findings
- The optimal tax progressivity parameter (`gamma`) for the calibrated U.S. economy is 0.0628.
- Implementing the optimal policy increases social welfare by 5.34% and reduces consumption and labor inequality.

### Counterfactual Analysis
- As income inequality rises, optimal policy recommendations include:
  - **Decreasing progressivity**: Encourages labor supply among high-wage individuals.
  - **Increasing transfers**: Mitigates consumption inequality for low-wage individuals.
- These adjustments balance equity and efficiency, improving aggregate labor supply and consumption.

---

## Limitations and Future Work
- **Model Simplifications**: Assumes no income effect in individual utility functions, which may affect behavioral predictions.
- **Static Weights**: Does not adjust weights for low-wage individuals as inequality rises.
- Future extensions could incorporate dynamic weighting and address the income effect for more robust policy recommendations.
