import time
from grove.grove_i2c_motor_driver import MotorDriver

motor = MotorDriver()

motor.set_dir(True)
motor.set_speed(255)

print("Motor spinning continuously. Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)   # keep process alive
except KeyboardInterrupt:
    motor.set_speed(0)
    print("Motor stopped")
