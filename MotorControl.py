import time
from grove.grove_i2c_motor_driver import MotorDriver

try:
    # Try to initialize. If it fails here, check your wiring!
    motor = MotorDriver() 
except Exception as e:
    print(f"Could not find motor driver: {e}")
    exit()

print("Motor driver connected. Starting loop...")

while True:
    try:
        motor.set_speed(100)
        motor.set_dir(True)
        time.sleep(2)

        motor.set_speed(70)
        motor.set_dir(False)
        time.sleep(2)
    except OSError:
        print("I2C Bus Error: Check your cables!")
        time.sleep(1) # Wait a bit before trying to reconnect