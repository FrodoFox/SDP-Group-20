import time
from grove.grove_i2c_motor_driver import MotorDriver

motor = MotorDriver()

try:
    while True:
        motor.StepperRun(200, 50)  
        time.sleep(0.1)
except KeyboardInterrupt:
    motor.stop()
    print("stopped")
