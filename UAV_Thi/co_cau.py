import sys
import time
import gpiod
from gpiod.line import Direction, Value

def servo(sig):
    # M? gpiochip
    with gpiod.request_lines(
        "/dev/gpiochip0",
        consumer="uav-servo",
        config={
            17: gpiod.LineSettings(
                direction=Direction.OUTPUT,
                output_value=Value.INACTIVE
            ),
            27: gpiod.LineSettings(
                direction=Direction.INPUT
            )
        }
    ) as lines:

        if sig == 1:
            # b?t output
            lines.set_value(17, Value.ACTIVE)

            # ch? input l�n 1
            while lines.get_value(27) == Value.INACTIVE:
                time.sleep(0.01)

            # t?t output
            lines.set_value(17, Value.INACTIVE)

if __name__ == "__main__":
    sig = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    servo(sig)