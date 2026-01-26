import time
from grove.grove_i2c_motor_driver import MotorDriver

motor = MotorDriver()

# Enable the stepper motor (Documentation for this sucks)
motor._MotorDriver__EnableStepper()

print("Stepper moving... Ctrl+C to stop.")

try:
    while True:
        # Stepernu(number_of_steps, direction)
        motor._MotorDriver__Stepernu(100, 0)
        time.sleep(0.5)
        
except KeyboardInterrupt:
    motor._MotorDriver__UnenableStepper()
    print("\nStopped and coils de-energized.")