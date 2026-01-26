import time
import sys
from grove.motor.i2c_motor_driver import I2CStepperMotor

RPM = 10 

# DEFINE PARAMS FOR THE MOTOR (using nema 23 stepper motor)
params = {
    'var-ratio': 1,         # Internal gear ratio
    'stride-angle': 1.8,    # Step angle of motor
    'rpm-max': 12,          # Maximum RPM of motor
    'sequences': [0b0001, 0b0011, 0b0010, 0b0110, 0b0100, 0b1100, 0b1000, 0b1001]   # Step sequence for stepper motors (don't change - unless you know what you're doing)
}

motor = I2CStepperMotor(params)

def move_stepper(degrees, direction):

    # Setting RPM - positive for clockwise and negative for anti-clockwise
    speed_rpm = RPM if direction == 1 else -RPM
    
    # CALCULATE EXPETED ROATION TIME
    expected_duration = (degrees / 360.0) / (RPM / 60.0)
    
    print(f"Moving {degrees} degrees {'Clockwise' if direction == 1 else 'Anti-clockwise'}...")
    
    try:

        # FORCING A CLEAN STARTING STATE (disabling and re-enabling motor)
        motor.enable(False)
        time.sleep(0.1)
        motor.enable(True)
        time.sleep(0.1)
        
        # SETTING THE SPEED AND STARTING ROTATIONS
        motor.speed(speed_rpm)
        time.sleep(0.1)         # buffering start to give the I2C bus time to settle
        motor.rotate(degrees)
        
        start_time = time.time()
        timeout = expected_duration + 2.0
        
        # WAITING FOR THE MOVEMENT TO FINISH
        while True:
            elapsed = time.time() - start_time

            try:
                left = motor.rotate()
                remaining = abs(left)
            except IOError:
                remaining = 999

            if (remaining < 0.1 and elapsed > 0.5) or (elapsed > timeout):
                break
            
            # PRINTING ANGLE UPDATE
            if remaining != 999:
                print(f"Angle remaining: {remaining:.2f}")
            
            # SLOWING DOWN POLLING TO KEEP I2C STABLE
            time.sleep(0.5)
            
        print("Movement complete.")
        
    except KeyboardInterrupt:
        print("\nEmergency Stop triggered.")
    finally:
        motor.enable(False) # always disable protect stepper motors (discharge coils)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        try:
            target_degrees = float(sys.argv[1])
            target_direction = int(sys.argv[2])
            if target_direction not in [0, 1]:
                print("Error: Direction must be 1 (CW) or 0 (CCW)")
            else:
                move_stepper(target_degrees, target_direction)
        except ValueError:
            print("Error: Please enter numbers for degrees and direction.")
    else:
        print("Usage: python MotorControl.py <degrees> <direction>")