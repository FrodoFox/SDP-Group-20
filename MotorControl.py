import time

from grove.grove_i2c_motor_driver import MotorDriver

motor = MotorDriver()

while True:

    # speed range: 0(lowest) - 100(fastest)

    motor.set_speed(100)

    # channel 1 only
    
    # to set channel 1&2: motor.set_speed(100, 100)
    
    # direction: True(clockwise), False(anti-clockwise)
    
    motor.set_dir(True)
    
    # channel 1 only,
    
    # to set channel 1&2: motor.set_dir(True, True)
    
    time.sleep(2)
    
    motor.set_speed(70)
    
    motor.set_dir(False)
    
    time.sleep(2)