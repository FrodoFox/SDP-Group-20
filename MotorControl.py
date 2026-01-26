import time
import sys
from grove.motor.i2c_motor_driver import I2CStepperMotor

RPM = 10 # Define a default RPM

# DEFINE PARAMS FOR THE MOTOR (using nema 23 stepper motor - different params available in documentation online)
params = {
    'var-ratio': 1,         # internal gear ratio
    'stride-angle': 1.8,    # step angle of motor
    'rpm-max': 12,          # maximum RPM of motor
    'sequences': [0b0001, 0b0011, 0b0010, 0b0110, 0b0100, 0b1100, 0b1000, 0b1001]   # Defining sequences of motor (8-step sequence for NEMA 23 - as well as the standard for stepper motors)
}

motor = I2CStepperMotor(params)

def move_stepper(degrees, direction):

    # Setting RPM - positive for clockwise and negative for anti-clockwise
    speed_rpm = RPM if direction == 1 else -RPM
    
    print(f"Moving {degrees} degrees {'Clockwise' if direction == 1 else 'Anti-clockwise'}...")
    
    try:
        # ENABLING AND SETTING SPEED BEFORE STARTING ROTATION (keep in this order to avoid bugs with the I2C motor library)
        motor.enable(True)
        motor.speed(speed_rpm)
        
        # TRIGGER THE ROTATION OF THE MOTOR
        motor.rotate(degrees)
        
        while True:

            # USING LIBRARIES .rotate() METHOD TO CHECK REMAINING ANGLE - LIBRARY NOTES IT'S UNSTABLE (but fuck it for now - can move to using rotation time later with RPM)
            try:
                left = motor.rotate() 
                remaining = abs(left)
            except IOError:

                # IF THE BUS GLITCHES IGNORE IT AND KEEP GOING.
                continue

            # If the register reports 0, the driver has finished its step sequence
            if remaining < 0.1:
                break
            
            # PRINTING ANGLE UPDATE
            print(f"Angle remaining: {remaining:.2f}")
            
            # SLOWING DOWN POLLING TO KEEP I2C STABLE - high-frequency polling can crash the I2C controller on the motor driver
            time.sleep(0.2)
            
        print("Movement complete.")
        
    except KeyboardInterrupt:
        print("\nEmergency Stop triggered.")
    finally:
        motor.enable(False) # always disable protect stepper motors (discharge coils to prevent overheat)

if __name__ == "__main__":

    # CHECKING ARGUMENTS GIVEN - sys.argv[1] = degrees, sys.argv[2] = direction
    if len(sys.argv) == 3:

        try:
            target_degrees = float(sys.argv[1])
            target_direction = int(sys.argv[2])
            
            # BASIC INPUT VALIDATION FOR DIRECTION OF MOTOR SPIN
            if target_direction not in [0, 1]:
                print("Error: Direction must be 1 (CW) or 0 (CCW)")
            else:
                move_stepper(target_degrees, target_direction)
                
        except ValueError:
            print("Error: Please enter numbers for degrees and direction.")
    else:
        print("Usage: python MotorControl.py <degrees> <direction>")