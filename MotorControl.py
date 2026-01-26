import time
from grove.motor.i2c_motor_driver import I2CStepperMotor

# Parameters for the stepper motor (using NEMA 23 (maybe??))
params = {
    'var-ratio': 1,         # Gear ratio
    'stride-angle': 1.8,    # Step angle
    'rpm-max': 12,          # Max speed
    'sequences': [0b0001, 0b0011, 0b0010, 0b0110, 0b0100, 0b1100, 0b1000, 0b1001] # (don't change these)
}

# Initialise the motor
motor = I2CStepperMotor(params)

# 3. Setup speed and movement
motor.speed(10)  # n RPM
motor.rotate(360) # Rotate 360 degrees

print("Starting Stepper...")
try:
    motor.enable(True) # This enables the motor and executes the movement
    
    while True:
        left = motor.rotate() # Checks how many degrees are left
        if left < 0.1:
            print("Movement complete.")
            break
        print(f"Angle remaining: {left:.2f}")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nEmergency Stop")
finally:
    motor.enable(False) # Disable the motor (needs done to prevent overheating)