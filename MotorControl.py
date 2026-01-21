import smbus
import time

# Constants for I2C Motor Driver V1.3
I2C_ADDR = 0x10  # Default I2C address for Motor Driver V1.3
STEPPER_MOTOR = 0x01  # Example motor ID (check your hardware docs)
CMD_SET_SPEED = 0x82
CMD_STEP = 0x1A

class StepperMotorDriver:
    def __init__(self, bus_num=1, address=I2C_ADDR):
        self.bus = smbus.SMBus(bus_num)
        self.address = address

    def set_speed(self, speed):
        """
        Set the speed of the stepper motor.
        :param speed: Speed value (0-255)
        """
        speed = max(0, min(255, speed))
        self.bus.write_i2c_block_data(self.address, CMD_SET_SPEED, [STEPPER_MOTOR, speed])

    def step(self, steps, direction=1):
        """
        Move the stepper motor a number of steps.
        :param steps: Number of steps to move
        :param direction: 1 for forward, 0 for backward
        """
        steps = max(0, min(255, steps))
        self.bus.write_i2c_block_data(self.address, CMD_STEP, [STEPPER_MOTOR, direction, steps])

    def stop(self):
        """
        Stop the stepper motor.
        """
        self.set_speed(0)

if __name__ == "__main__":
    motor = StepperMotorDriver()
    motor.set_speed(100)
    motor.step(50, direction=1)
    time.sleep(1)
    motor.stop()