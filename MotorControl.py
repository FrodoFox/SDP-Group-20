import time
import sys
from grove.motor.i2c_motor_driver import I2CStepperMotor

RPM = 10 # Define a default RPM

# DEFINE PARAMS FOR THE MOTOR (using nema 23 stepper motor - different params available in documentation online)
params = {
    'var-ratio': 1,         # Internal gear ratio
    'stride-angle': 1.8,    # Step angle of motor
    'rpm-max': 12,          # Maximum RPM of motor
    'sequences': [0b0001, 0b0011, 0b0010, 0b0110, 0b0100, 0b1100, 0b1000, 0b1001]   # Defining sequences of motor (8-step sequence for NEMA 23 - as well as the standard for stepper motors)
}

motor = I2CStepperMotor(params)

def move_stepper(degrees, direction):

    # Setting RPM - positive for clockwise and negative for anti-clockwise
    speed_rpm = RPM if direction == 1 else -RPM
    
    print(f"Moving {degrees} degrees {'Clockwise' if direction == 1 else 'Anti-clockwise'}...")
    
    try:
        motor.speed(speed_rpm)
        motor.rotate(degrees)
        motor.enable(True)
        
        # WAITING FOR THE MOVEMENT TO FINISH
        while True:
            left = motor.rotate() # get remaining number of degrees to rotate
            if left < 0.1:
                break
            print(f"Angle remaining: {left:.2f}")
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
            
            # basic input validation for direction of rotation
            if target_direction not in [0, 1]:
                print("Error: Direction must be 1 (CW) or 0 (CCW)")
            else:
                move_stepper(target_degrees, target_direction)
                
        except ValueError:
            print("Error: Please enter numbers for degrees and direction.")
    else:
        print("Usage: python MotorControl.py <degrees> <direction>")