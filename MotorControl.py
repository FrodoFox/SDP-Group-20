import time
from grove.motor.i2c_motor_driver import MotorDriver, I2CStepperMotor

params = {
    'var-ratio': 64,          # Gear ratio
    'stride-angle': 5.625,    # Step angle
    'rpm-max': 12,            # Max speed
    'sequences': [0b0001, 0b0011, 0b0010, 0b0110, 0b0100, 0b1100, 0b1000, 0b1001]
}

# 2. Initialize the correct class
motor = I2CStepperMotor(params)

# 3. Setup speed and movement
motor.speed(10)  # 10 RPM
motor.rotate(360) # Prepare to rotate 360 degrees

print("Starting Stepper...")
try:
    motor.enable(True) # This sends the Enable and Sequence commands
    
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
    motor.enable(False) # De-energize coils