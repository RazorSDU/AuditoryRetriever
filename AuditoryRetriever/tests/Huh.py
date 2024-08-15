import pygetwindow as gw
import keyboard
import mouse
import time
import json
import win32api
import win32con
import win32gui

# Variables to track the state and recording details
is_recording = False
is_playing = False
macro_startup_data = []
macro_repeat_data = []
macro_startup_file = "Macro_Startup.txt"
macro_repeat_file = "Macro_Repeat.txt"
start_time = None
recording_macro_type = None

# Function to record a mouse click
def record_click(event):
    global start_time
    if is_recording and isinstance(event, mouse.ButtonEvent) and event.event_type == 'down':
        current_time = time.time()
        if start_time is None:
            start_time = current_time

        if recording_macro_type == "startup":
            macro_data = macro_startup_data
        else:
            macro_data = macro_repeat_data

        if macro_data:
            last_time = macro_data[-1]["time"]
            time_diff = current_time - (start_time + last_time)
        else:
            time_diff = 0

        # Get global mouse position
        x, y = win32api.GetCursorPos()

        # Get window position
        window = gw.getWindowsWithTitle("SM-A546B")
        if not window:
            print("Window not found.")
            return

        window = window[0]
        hwnd = window._hWnd

        # Get client area coordinates
        client_rect = win32gui.GetClientRect(hwnd)
        client_point = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))
        client_x, client_y = client_point

        # Calculate relative position
        rel_x = x - client_x
        rel_y = y - client_y

        macro_data.append({"time": current_time - start_time, "x": rel_x, "y": rel_y, "delay": time_diff})
        print(f"Recorded click at ({rel_x}, {rel_y}) relative to window with delay {time_diff} seconds")

# Function to start recording
def start_recording(macro_type):
    global is_recording, start_time, recording_macro_type
    is_recording = True
    start_time = time.time()
    recording_macro_type = macro_type
    if macro_type == "startup":
        macro_startup_data.clear()
    else:
        macro_repeat_data.clear()
    print(f"Recording {macro_type} macro started...")

# Function to stop recording and save to file
def stop_recording():
    global is_recording
    is_recording = False
    if recording_macro_type == "startup":
        with open(macro_startup_file, 'w') as f:
            json.dump(macro_startup_data, f)
        print("Startup macro recording stopped and saved to Macro_Startup.txt")
    else:
        with open(macro_repeat_file, 'w') as f:
            json.dump(macro_repeat_data, f)
        print("Repeat macro recording stopped and saved to Macro_Repeat.txt")

# Function to load macros from files
def load_macros():
    global macro_startup_data, macro_repeat_data
    with open(macro_startup_file, 'r') as f:
        macro_startup_data = json.load(f)
    with open(macro_repeat_file, 'r') as f:
        macro_repeat_data = json.load(f)
    print("Macros loaded from files")

# Function to simulate a mouse click at a specific position in a window
def click_in_window(hwnd, x, y):
    lParam = win32api.MAKELONG(x, y)
    win32api.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lParam)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
    win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)
    show_click_marker(hwnd, x, y)

# Function to show a visual marker at the click position
def show_click_marker(hwnd, x, y, duration=0.2):
    hdc = win32gui.GetDC(hwnd)
    brush = win32gui.CreateSolidBrush(win32api.RGB(255, 0, 0))
    pen = win32gui.CreatePen(win32con.PS_SOLID, 1, win32api.RGB(255, 0, 0))
    old_brush = win32gui.SelectObject(hdc, brush)
    old_pen = win32gui.SelectObject(hdc, pen)

    win32gui.Ellipse(hdc, x-5, y-5, x+5, y+5)
    win32gui.SelectObject(hdc, old_brush)
    win32gui.SelectObject(hdc, old_pen)
    win32gui.DeleteObject(brush)
    win32gui.DeleteObject(pen)
    win32gui.ReleaseDC(hwnd, hdc)

    time.sleep(duration)

    # Clear the marker by redrawing with background color
    hdc = win32gui.GetDC(hwnd)
    brush = win32gui.CreateSolidBrush(win32gui.GetBkColor(hdc))
    pen = win32gui.CreatePen(win32con.PS_SOLID, 1, win32gui.GetBkColor(hdc))
    old_brush = win32gui.SelectObject(hdc, brush)
    old_pen = win32gui.SelectObject(hdc, pen)

    win32gui.Ellipse(hdc, x-5, y-5, x+5, y+5)
    win32gui.SelectObject(hdc, old_brush)
    win32gui.SelectObject(hdc, old_pen)
    win32gui.DeleteObject(brush)
    win32gui.DeleteObject(pen)
    win32gui.ReleaseDC(hwnd, hdc)

# Function to play a macro
def play_macro(macro_data):
    # Get the window handle
    window = gw.getWindowsWithTitle("SM-A546B")
    if not window:
        print("Window not found.")
        return

    window = window[0]
    hwnd = window._hWnd

    for action in macro_data:
        time.sleep(action["delay"])
        # Calculate relative position
        rel_x = action["x"]
        rel_y = action["y"]
        click_in_window(hwnd, rel_x, rel_y)

# Function to play the macros once
def play_macros_once():
    load_macros()
    play_macro(macro_startup_data)
    play_macro(macro_repeat_data)

# Function to start playing macros in loop
def start_loop():
    global is_playing
    is_playing = True
    print("Started playing macros in loop...")
    load_macros()
    play_macro(macro_startup_data)
    while is_playing:
        play_macro(macro_repeat_data)

# Function to stop playing macros in loop
def stop_loop():
    global is_playing
    is_playing = False
    print("Stopped playing macros in loop.")

# Set up the keyboard hotkeys
keyboard.add_hotkey('e', lambda: start_recording("startup") if not is_recording else stop_recording())
keyboard.add_hotkey('r', lambda: start_recording("repeat") if not is_recording else stop_recording())
keyboard.add_hotkey('a', play_macros_once)
keyboard.add_hotkey('s', lambda: start_loop() if not is_playing else stop_loop())

# Set up the mouse event listener
mouse.hook(record_click)

print("Press 'E' to start/stop recording the startup macro. Press 'R' to start/stop recording the repeat macro. Press 'A' to play the macros once. Press 'S' to start/stop playing the macros in loop. Press 'Esc' to exit.")

try:
    while True:
        if keyboard.is_pressed('esc'):  # Exit the loop on pressing 'esc'
            break
        time.sleep(0.001)

except KeyboardInterrupt:
    print("\nProgram terminated.")
