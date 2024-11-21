
import matplotlib.pyplot as plt
import numpy as np
import control as ctrl
from scipy.optimize import fsolve

# Define poles and zeros
poles = [0, -4, -2+4j, -2-4j]  # s = 0, s = -3, s = -1 + j, s = -1 - j
zeros = []  # No zeros in this transfer function

# Create the transfer function
numerator_coeffs = np.poly(zeros) if zeros else [1]  # If no zeros, use [1]
denominator_coeffs = np.poly(poles)
system = ctrl.TransferFunction(numerator_coeffs, denominator_coeffs)

# Generate the root locus plot
fig, ax = plt.subplots()
ctrl.rlocus(system, ax=ax)

# Calculate centroid of the asymptotes
num_poles = len(poles)
num_zeros = len(zeros)
num_asymptotes = num_poles - num_zeros
real_parts_of_poles = [np.real(p) for p in poles]
real_parts_of_zeros = [np.real(z) for z in zeros]
centroid = np.round((sum(real_parts_of_poles) - sum(real_parts_of_zeros)) / num_asymptotes, 3)

# Calculate asymptote angles
asymptote_angles = [np.round((2 * k + 1) * 180 / num_asymptotes, 3) for k in range(num_asymptotes)]

# Print centroid and asymptote angles
print(f"Centroid: {centroid}")
print(f"Asymptote Angles: {asymptote_angles}")

# Mark centroid on the plot
ax.plot(centroid, 0, 'go', label="Centroid")

# Plot the asymptotes
for angle in asymptote_angles:
    angle_rad = np.deg2rad(angle)  # Convert angle to radians
    x_end = np.cos(angle_rad)
    y_end = np.sin(angle_rad)
    ax.plot([centroid, centroid + 10 * x_end], [0, 10 * y_end], 'r--', label=f"Asymptote at {angle}°")
    ax.text(centroid + 8 * x_end, 8 * y_end, f"{angle}°", fontsize=10, color='red')

# Function to find the derivative of the characteristic equation
def characteristic_eq(s):
    return np.polyval(denominator_coeffs, s)

# Function to compute the derivative of the denominator
def derivative_characteristic_eq(s):
    return np.polyder(denominator_coeffs)

# Find the breakaway points by solving for roots of the derivative of the characteristic equation
def find_breakaway_points():
    derivative_coeffs = derivative_characteristic_eq(0)  # Get the coefficients of the derivative
    breakaway_points = np.roots(derivative_coeffs)  # Find roots of the derivative polynomial
    return breakaway_points

# Function to evaluate K for a given s value
def evaluate_K(s):
    # Evaluate the characteristic equation at s and return K
    D_s = np.polyval(denominator_coeffs, s)
    N_s = np.polyval(numerator_coeffs, s)
    K = abs(N_s / D_s)  # Gain (K)
    return K

# Calculate breakaway points and check for K > 0
breakaway_points = find_breakaway_points()
valid_breakaway_points = []

for bp in breakaway_points:
    bp_rounded = np.round(bp, 3)  # Round breakaway points to 3 decimal places
    K_value = evaluate_K(bp_rounded)  # Calculate gain at the breakaway point
    if K_value > 0:  # Check if K > 0 for both real and complex points
        valid_breakaway_points.append(bp_rounded)

# Print valid breakaway points
print(f"Valid Breakaway Points: {valid_breakaway_points}")

# Mark valid breakaway points on the plot
for bp in valid_breakaway_points:
    ax.plot(bp.real, bp.imag, 'ro', label="Valid Breakaway Point")

# Calculate and mark angle of departure (for complex poles)
def angle_of_departure(pole):
    total_angle = 180  # Initial angle (180 degrees)
    for other_pole in poles:
        if other_pole != pole:
            angle = np.angle(pole - other_pole, deg=True)  # Calculate the angle of departure
            total_angle -= angle
    return np.round(total_angle, 3)

# Calculate and print angle of departure for complex poles
for pole in poles:
    if np.imag(pole) != 0:  # Check if it's a complex pole
        departure_angle = angle_of_departure(pole)
        print(f"Angle of Departure from pole {pole}: {departure_angle}°")

        # Plot angle of departure
        ax.plot([pole.real, pole.real + 2 * np.cos(np.deg2rad(departure_angle))],
                [pole.imag, pole.imag + 2 * np.sin(np.deg2rad(departure_angle))],
                'b-', label=f"Angle of Departure from {pole}")

# Add labels and grid to the plot
plt.title("Root Locus with Asymptotes, Breakaway Points, Angle of Departure")
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.show()
