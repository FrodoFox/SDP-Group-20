import time
from grove.grove_i2c_motor_driver import MotorDriver

motor = MotorDriver()

print("Moving stepper... Ctrl+C to stop.")

try:
    while True:
        motor.set_speed(255, 255) # Sets speed for both M1 and M2
        time.sleep(1)
        motor.stop()
        break
        
except KeyboardInterrupt:
    motor.stop()
    print("Stopped")