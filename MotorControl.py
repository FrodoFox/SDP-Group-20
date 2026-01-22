import time
from grove.grove_i2c_motor_driver import MotorDriver

motor = MotorDriver()

<<<<<<< HEAD
motor.set_dir(True)
motor.set_speed(255)

print("Motor spinning continuously. Ctrl+C to stop.")

=======
>>>>>>> 69c4fa7129d95c2d746c9acc2b57d97a54254af6
try:
    while True:
        motor.StepperRun(200, 50)  
        time.sleep(0.1)
except KeyboardInterrupt:
    motor.stop()
    print("stopped")
