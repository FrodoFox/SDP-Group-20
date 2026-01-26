import time
import sys
from grove.motor.i2c_motor_driver import I2CStepperMotor

RPM = 10 

# DEFINE PARAMS FOR THE MOTOR
params = {
    'var-ratio': 1,
    'stride-angle': 1.8,
    'rpm-max': 12,
    'sequences': [0b0001, 0b0011, 0b0010, 0b0110, 0b0100, 0b1100, 0b1000, 0b1001]
}

motor = I2CStepperMotor(params)

def move_stepper(degrees, direction):
    speed_rpm = RPM if direction == 1 else -RPM
    
    # MATH: Calculate how long it SHOULD take. 
    # (degrees/360) is fraction of a turn. (RPM/60) is revs per second.
    expected_duration = (degrees / 360.0) / (RPM / 60.0)
    
    print(f"Moving {degrees} degrees {'Clockwise' if direction == 1 else 'Anti-clockwise'}...")
    print(f"Estimated time: {expected_duration:.2f}s")
    
    try:
        motor.enable(True)
        motor.speed(speed_rpm)
        
        # Start the rotation
        motor.rotate(degrees)
        
        start_time = time.time()
        # Give it a 20% "safety buffer" over the calculated time
        timeout = expected_duration * 1.2 
        
        while True:
            elapsed = time.time() - start_time
            
            # 1. READ HARDWARE: The library's rotate() getter
            # We wrap this to catch the "unstable interface" issues mentioned in the lib
            try:
                left = motor.rotate()
                remaining = abs(left)
            except:
                remaining = degrees # Fallback if I2C fails mid-read
            
            # 2. EXIT CONDITION A: Hardware says we are done
            # We only trust the hardware 'done' signal if at least some time has passed
            if remaining < 0.5 and elapsed > 0.5:
                break
                
            # 3. EXIT CONDITION B: Safety Timeout
            # If the I2C bus is 'unstable' and never reports 0, we stop based on time.
            if elapsed > timeout:
                print("Safety timeout reached (Time-based stop).")
                break
            
            print(f"Angle remaining: {remaining:.2f} | Time elapsed: {elapsed:.1f}s")
            time.sleep(0.3)
            
        print("Movement complete.")
        
    except KeyboardInterrupt:
        print("\nEmergency Stop triggered.")
    finally:
        motor.enable(False)

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