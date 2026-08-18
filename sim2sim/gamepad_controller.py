import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Use a virtual video driver to avoid display detection issues.
import pygame
import time

class GamepadController:
    def __init__(self, deadzone=0.15):
        # Initialize only the display and joystick subsystems.
        pygame.display.init()
        pygame.joystick.init()
        
        self.joystick = None
        self.deadzone = deadzone

        # Initialize axis and button state dictionaries.
        self.axes = {}
        self.buttons = {}

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[INFO] Gamepad connected: {self.joystick.get_name()}")
            for i in range(self.joystick.get_numaxes()): self.axes[i] = 0.0
            for i in range(self.joystick.get_numbuttons()): self.buttons[i] = False
        else:
            print("[WARNING] No gamepad detected.")

    def update(self):
        """Update all button and joystick states from the event queue."""
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.axes[event.axis] = event.value
            elif event.type == pygame.JOYBUTTONDOWN:
                self.buttons[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                self.buttons[event.button] = False

    def get_commands(self):
        """Read and convert joystick input into velocity commands."""
        self.update()
        if self.joystick is None: return 0.0, 0.0, 0.0

        # Match the configured gamepad hardware mapping.
        lx = self.axes.get(0, 0.0)  # Left stick horizontal (axis 0).
        ly = self.axes.get(1, 0.0)  # Left stick vertical (axis 1).
        
        # Steering uses the measured right-stick axis.
        rx = self.axes.get(2, 0.0)  

        lx = 0.0 if abs(lx) < self.deadzone else lx
        ly = 0.0 if abs(ly) < self.deadzone else ly
        rx = 0.0 if abs(rx) < self.deadzone else rx

        # Positive x is forward, positive y is left, and positive yaw is counter-clockwise.
        cmd_x = -ly * 1.0  
        cmd_y = -lx * 0.5  
        cmd_yaw = -rx * 1.0 

        return cmd_x, cmd_y, cmd_yaw

    # ==========================================
    # Button accessors matching the configured gamepad layout.
    # ==========================================
    def get_button_a(self): return self.buttons.get(0, False)
    def get_button_b(self): return self.buttons.get(1, False)
    def get_button_x(self): return self.buttons.get(3, False)
    def get_button_y(self): return self.buttons.get(4, False)
    def get_button_lb(self): return self.buttons.get(6, False)
    def get_button_rb(self): return self.buttons.get(7, False)
    
    # LT/RT are exposed as digital buttons on this controller.
    def get_button_lt(self): return self.buttons.get(8, False) 
    def get_button_rt(self): return self.buttons.get(9, False)
    
    def get_button_back(self): return self.buttons.get(10, False)
    def get_button_start(self): return self.buttons.get(11, False)

# ==========================================
# Hardware detection utility with joystick-axis diagnostics.
# ==========================================
if __name__ == "__main__":
    print("--- Gamepad hardware monitor ---")
    print("Move the left and right sticks separately and watch the axis values.")
    print("Press Ctrl+C to exit.\n")
    
    # Disable the deadzone to expose raw axis values.
    pad = GamepadController(deadzone=0.0) 
    try:
        while True:
            pad.update()
            
            # Collect currently pressed button IDs.
            pressed_btns = [k for k, v in pad.buttons.items() if v]
            
            # Show active axes while filtering small stick drift.
            active_axes = {k: round(v, 2) for k, v in pad.axes.items() if abs(v) > 0.05}
            
            print(f"\r[Monitor] Buttons: {pressed_btns} | Axes: {active_axes}                        ", end="")
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n[INFO] Hardware monitor stopped.")
        pygame.quit()
