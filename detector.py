import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(_file_))
CALIB_FILE = os.path.join(SCRIPT_DIR, "calibration.json")

import cv2
import numpy as np
import time
from PIL import Image
from util import get_limits

DEFAULT_CALIB_WORD = "SOS"


# Morse code dictionary (morse -> char)
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',
    '----.': '9'
}

# Build reverse mapping (char -> morse)
CHAR_TO_MORSE = {v: k for k, v in MORSE_CODE.items()}

# Default (initial) thresholds (will be tuned by calibration)
DOT_THRESHOLD = 0.25
DASH_THRESHOLD = 0.65
GAP_THRESHOLD = 2 * DOT_THRESHOLD
WORD_GAP_THRESHOLD = 7 * DOT_THRESHOLD
DEBOUNCE_FRAMES = 2

# Utility: convert a word (string) -> sequence of '.' and '-' for its symbols
def word_to_on_labels(word):
    word = word.upper()
    labels = []
    for ch in word:
        if ch == " ":
            continue
        if ch not in CHAR_TO_MORSE:
            raise ValueError(f"Character '{ch}' not supported in Morse mapping.")
        morse = CHAR_TO_MORSE[ch]
        for s in morse:
            labels.append(s)  # '.' or '-'
    return labels

# Decode from the recorded durations using thresholds
def decode_from_durations(on_durations, off_durations, thresholds):
    # thresholds: dict with 'dot_dash_cutoff', 'gap', 'word_gap'
    dot_dash_cutoff = thresholds['dot_dash_cutoff']
    gap_threshold = thresholds['gap']
    word_gap_threshold = thresholds['word_gap']
    
    morse_symbols = []
    decoded_message = ""
    current_symbol = ""
    
    n = len(on_durations)
    for i in range(n):
        on_d = on_durations[i]
        # classify on duration
        if on_d < dot_dash_cutoff:
            current_symbol += "."
        else:
            current_symbol += "-"
        # determine gap after (if any)
        if i < len(off_durations):
            off_d = off_durations[i]
            if off_d > word_gap_threshold:
                # end of letter then space
                morse_symbols.append(current_symbol)
                current_symbol = ""
                morse_symbols.append(" ")  # word separator marker
            elif off_d > gap_threshold:
                # end of letter
                morse_symbols.append(current_symbol)
                current_symbol = ""
            else:
                # within-letter gap (do nothing)
                pass
    # append last symbol if exists
    if current_symbol:
        morse_symbols.append(current_symbol)
    
    # convert morse_symbols -> text
    current_morse = ""
    for sym in morse_symbols:
        if sym == " ":
            # word break
            if current_morse:
                decoded_message += MORSE_CODE.get(current_morse, "?")
                current_morse = ""
            decoded_message += " "
        else:
            # a morse chunk (like '...' or '-.-')
            decoded_message += MORSE_CODE.get(sym, "?")
    # If any trailing morse chunk left unconverted (rare)
    return decoded_message.strip()

# Compute thresholds from labeled on_durations (with labels '.'/'-') and off_durations
def estimate_thresholds_from_samples(all_on_durations, all_on_labels, all_off_durations):
    # all_on_durations: list/np.array
    # all_on_labels: list same length, each '.' or '-'
    on = np.array(all_on_durations)
    labels = np.array(all_on_labels)
    
    dot_durs = on[labels == '.'] if np.any(labels == '.') else np.array([])
    dash_durs = on[labels == '-'] if np.any(labels == '-') else np.array([])
    
    # Fallback defaults if missing
    if dot_durs.size == 0 and dash_durs.size == 0:
        dot_dash_cutoff = DOT_THRESHOLD
    elif dot_durs.size == 0:  # only dash samples
        dash_med = np.median(dash_durs)
        dot_dash_cutoff = dash_med * 0.6
    elif dash_durs.size == 0:  # only dot samples
        dot_med = np.median(dot_durs)
        dot_dash_cutoff = dot_med * 1.8
    else:
        dot_med = np.median(dot_durs)
        dash_med = np.median(dash_durs)
        dot_dash_cutoff = (dot_med + dash_med) / 2.0
    
    # Off durations: cluster by percentiles to decide letter/word gaps
    off = np.array(all_off_durations) if len(all_off_durations) > 0 else np.array([])
    if off.size == 0:
        gap_threshold = GAP_THRESHOLD
        word_gap_threshold = WORD_GAP_THRESHOLD
    else:
        # Small gaps will occupy lower percentiles; letter gaps around middle; word gaps high
        # Use percentiles to separate
        p40 = np.percentile(off, 40)
        p85 = np.percentile(off, 85)
        # make sure thresholds are sensible (not smaller than dot/dash)
        gap_threshold = max(p40, dot_dash_cutoff * 0.8)
        word_gap_threshold = max(p85, gap_threshold * 2.0)
    
    # Return thresholds
    return {
        'dot_dash_cutoff': float(dot_dash_cutoff),
        'gap': float(gap_threshold),
        'word_gap': float(word_gap_threshold)
    }

# Calibration routine: automatic, iterative, numpy-only
def calibrate(cap, expected_word, max_iterations=10, verbose=True):
    """
    Runs camera-based calibration. The user should flash the LED for the given expected_word.
    The function will collect ON/OFF durations, compute thresholds, decode and check.
    If decoded != expected_word, it will accumulate more data and retry up to max_iterations.
    """
    expected_word = expected_word.strip().upper()
    if not expected_word:
        raise ValueError("Expected word must be non-empty.")
    # Build expected on-label sequence (only '.' and '-')
    expected_on_labels = word_to_on_labels(expected_word)
    needed_symbols = len(expected_on_labels)
    if verbose:
        print(f"Calibrating for word: '{expected_word}' -> {needed_symbols} symbols.")
        print("Flash the LED to send the calibration word repeatedly when camera view appears.")
        print("Press 'q' to abort calibration and exit.")
    
    # We will accumulate samples across iterations to get robust statistics
    cumulative_on = []
    cumulative_on_labels = []
    cumulative_off = []
    
    # Debounce history and state variables (for use inside calibration)
    detection_history = []
    light_visible = False
    visibility_start_time = None
    last_visible_time = None
    
    # show window indicating calibration active
    while True:
        iteration = 0
        success = False
        
        while iteration < max_iterations and not success:
            iteration += 1
            if verbose:
                print(f"\nCalibration pass #{iteration}: waiting to collect one full transmission...")
            # For each pass, we collect one sequence of on/off durations that covers at least the needed_symbols
            on_durations = []
            off_durations = []
            on_labels = []
            detection_history = []
            light_visible = False
            visibility_start_time = None
            last_visible_time = None
            
            start_pass_time = time.time()
            pass_timeout = 20.0  # seconds to give user to flash the whole word per pass
            collected_symbols = 0
            last_loop_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Camera read failed during calibration.")
                
                # Process frame for green detection (same as main)
                hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lowerLimit, upperLimit = get_limits(color=[0,255,0])
                mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
                mask_ = Image.fromarray(mask)
                bbox = mask_.getbbox()
                
                current_time = time.time()
                light_detected = bbox is not None
                
                # Debounce
                detection_history.append(light_detected)
                if len(detection_history) > DEBOUNCE_FRAMES:
                    detection_history.pop(0)
                debounced_detection = all(detection_history) if len(detection_history) == DEBOUNCE_FRAMES else light_detected
                
                # Track visibility times
                if debounced_detection:
                    if not light_visible:
                        # just became visible
                        light_visible = True
                        visibility_start_time = current_time
                    last_visible_time = current_time
                else:
                    if light_visible:
                        # just became invisible -> we have an ON duration
                        duration_on = current_time - visibility_start_time
                        on_durations.append(duration_on)
                        # label will be assigned using expected_on_labels based on count
                        if collected_symbols < needed_symbols:
                            on_labels.append(expected_on_labels[collected_symbols])
                        else:
                            # If we got extra symbol, label it as unknown - but still store
                            on_labels.append('.')
                        collected_symbols += 1
                        # start off-duration timer by storing last_visible_time and continuing
                        visibility_start_time = None
                        light_visible = False
                # For off durations: we measure gap between last_visible_time and next visible start.
                # We approximate off durations by measuring time between end of last ON and start of next ON.
                # Implementation: when light becomes visible, if last_visible_time exists we can compute off duration.
                if debounced_detection and not light_visible:
                    # handled above, skip
                    pass
                
                # Compute off durations by checking transitions: if currently visible and we have record of last off start
                # Simpler: track last_off_start when light becomes invisible; when it becomes visible again, duration = now - last_off_start
                # We'll implement small state machine for that:
                # (We can piggyback on variables defined above.)
                
                # (Implement off-duration measurement)
                # We use 'last_invisible_time' variable to mark start of off; create if not present.
                if 'last_invisible_time' not in locals():
                    last_invisible_time = None
                # If light is not visible and last_invisible_time is None -> mark start
                if not debounced_detection and last_invisible_time is None:
                    last_invisible_time = current_time
                # If light becomes visible and last_invisible_time is set -> record off duration
                if debounced_detection and last_invisible_time is not None:
                    off_dur = current_time - last_invisible_time
                    # Only record off_durations if we have previously recorded at least one ON (so this is gap after an on)
                    if len(on_durations) > 0:
                        off_durations.append(off_dur)
                    last_invisible_time = None
                
                # Show some UI on frame for calibration
                # Draw bounding box if detected
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
                    green_region = hsvImage[y1:y2, x1:x2]
                    avg_value = np.mean(green_region[:, :, 2]) if green_region.size > 0 else 0
                    cv2.putText(frame, f"Brightness: {int(avg_value)}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                # Show counts and live interpreted morse for the CURRENT pass
                # Interpret on_durations collected so far using a temporary cutoff (use current estimates or defaults)
                # For the first pass, until we have a better cutoff, use DOT_THRESHOLD/DASH_THRESHOLD
                temp_cutoff = (DOT_THRESHOLD + DASH_THRESHOLD) / 2.0
                temp_morse_live = ""
                for d in on_durations:
                    temp_morse_live += '.' if d < temp_cutoff else '-'
                cv2.putText(frame, f"Calibration PASS #{iteration}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(frame, f"Collected symbols: {collected_symbols}/{needed_symbols}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"Live: {temp_morse_live}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                cv2.putText(frame, f"Expected (ON sequence): {''.join(expected_on_labels)}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
                
                cv2.imshow('Calibration - Flash the word', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    # Abort calibration entirely
                    cv2.destroyWindow('Calibration - Flash the word')
                    raise KeyboardInterrupt("Calibration aborted by user (q).")
                
                # If we've collected at least the needed number of on symbols, we can end this pass
                if collected_symbols >= needed_symbols:
                    # Give a small pause so off_durations that immediately follow are recorded
                    # (user might leave a long final off gap that we also want)
                    # Wait a short fixed interval to capture trailing off duration
                    end_wait_start = time.time()
                    while time.time() - end_wait_start < 0.4:
                        ret2, frame2 = cap.read()
                        if not ret2:
                            break
                        hsvImage2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
                        mask2 = cv2.inRange(hsvImage2, lowerLimit, upperLimit)
                        mask2_ = Image.fromarray(mask2)
                        bbox2 = mask2_.getbbox()
                        current_time2 = time.time()
                        # manage last_invisible_time similarly
                        if bbox2 is None and last_invisible_time is None:
                            last_invisible_time = current_time2
                        if bbox2 is not None and last_invisible_time is not None:
                            off_dur = current_time2 - last_invisible_time
                            if len(on_durations) > 0 and (len(off_durations) < len(on_durations)):
                                off_durations.append(off_dur)
                            last_invisible_time = None
                        cv2.imshow('Calibration - Flash the word', frame2)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyWindow('Calibration - Flash the word')
                            raise KeyboardInterrupt("Calibration aborted by user (q).")
                    break
                
                # timeout per pass
                if time.time() - start_pass_time > pass_timeout:
                    if verbose:
                        print("Calibration pass timeout: no full transmission detected. Retrying pass...")
                    break
            
            # If pass collected no symbols, retry
            if len(on_durations) == 0:
                if verbose:
                    print("No symbols collected in this pass; try again.")
                continue
            
            # Map collected on labels: we already assigned expected_on_labels in order; if extra collected, we appended '.' placeholder.
            # Append collected samples to cumulative arrays
            cumulative_on.extend(on_durations)
            cumulative_on_labels.extend(on_labels)
            cumulative_off.extend(off_durations)
            
            # Estimate thresholds using cumulative samples
            thresholds = estimate_thresholds_from_samples(cumulative_on, cumulative_on_labels, cumulative_off)
            if verbose:
                print("Estimated thresholds:", thresholds)
            
            # Try decoding the recorded sequence from this pass (or use whole cumulative sequence)
            # We'll test decoding on the MOST RECENT pass (on_durations/off_durations)
            test_decoded = decode_from_durations(on_durations, off_durations, thresholds)
            # Normalize spaces and uppercase for comparison
            test_comp = test_decoded.replace(" ", "").upper()
            expected_comp = expected_word.replace(" ", "").upper()
            if verbose:
                print(f"Decoded this pass (no spaces): '{test_comp}'  | expected: '{expected_comp}'")
                print(f"Decoded full string (with spaces): '{test_decoded}'")
            
            if test_comp == expected_comp:
                if verbose:
                    print("Calibration pass decoded correctly. Finalizing thresholds.")
                success = True
                # Optionally show final thresholds
                return thresholds
            else:
                # Not yet correct: continue and accumulate more samples
                if verbose:
                    print("Decoded word did not match expected. Accumulating more data and retrying...")
                # continue to next iteration which will collect more samples
        # If here: either success returned or max_iterations exhausted
        if not success:
            # As a last attempt, try decoding using the cumulative collection (maybe different pass alignment)
            thresholds = estimate_thresholds_from_samples(cumulative_on, cumulative_on_labels, cumulative_off)
            # Try decode using cumulative sequences grouped in sliding windows equivalent to expected count
            # Build a combined test by sliding over cumulative_on to find a subsequence matching the expected length
            cum_on = np.array(cumulative_on)
            cum_off = np.array(cumulative_off) if len(cumulative_off) > 0 else np.array([])
            found = False
            for start in range(0, max(1, len(cum_on) - needed_symbols + 1)):
                sub_on = cum_on[start:start+needed_symbols].tolist()
                # sub_off length should be at least needed_symbols-1; take accordingly
                sub_off = cum_off[start:start+needed_symbols-1].tolist() if cum_off.size >= max(0, needed_symbols-1) else cum_off.tolist()
                test_decoded = decode_from_durations(sub_on, sub_off, thresholds)
                if test_decoded.replace(" ", "").upper() == expected_comp:
                    found = True
                    if verbose:
                        print("Found matching subsequence in cumulative data.")
                    return thresholds
            if not found:
                # If still not found, either ask user to re-run script (but per instruction we should iterate automatically)
                # We'll loop back and keep trying until user interrupts. But to avoid infinite loops, we'll inform user.
                print("\nUnable to calibrate automatically with current samples.")
                print("Make sure you flash the calibration word cleanly (use clear DOTs and DASHes).")
                print("We'll continue trying; press 'q' in the window to abort.")
                # continue outer while True to keep trying until user aborts or until success
                # Note: because we're not saving thresholds, we'll keep accumulating; user can re-run script if desired.
                # Loop will go to next iteration and keep collecting more samples.
                continue


def save_calibration(thresholds):
    try:
        with open(CALIB_FILE, "w") as f:
            json.dump(thresholds, f, indent=4)
        print(f"\nCalibration saved to {CALIB_FILE}\n")
    except Exception as e:
        print("Error saving calibration:", e)


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        return None
    try:
        with open(CALIB_FILE, "r") as f:
            data = json.load(f)
        print(f"\nLoaded calibration from {CALIB_FILE}:\n{data}\n")
        return data
    except Exception as e:
        print("Error loading calibration:", e)
        return None


# ------------ Main detection program (after calibration) -------------
def run_main_detector():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Try loading saved calibration thresholds
    thresholds = load_calibration()
    print("[DEBUG] thresholds after load:", thresholds)

    # If no saved calibration, run calibration first (only once)
    if thresholds is None:
        print("No calibration file found. Starting calibration.\n")
        calib_word = input("Enter calibration word (example: SOS): ").strip()
        print("Starting calibration... Please flash the word.")
        thresholds = calibrate(cap, calib_word, max_iterations=12, verbose=True)
        save_calibration(thresholds)

    # Unpack thresholds for main loop use
    dot_dash_cutoff = thresholds['dot_dash_cutoff']
    gap_threshold = thresholds['gap']
    word_gap_threshold = thresholds['word_gap']

    try:
        # Now run the main continuous detection using calibrated thresholds
        current_morse = ""
        decoded_message = ""
        detection_history = []
        light_visible = False
        visibility_start_time = None
        last_visible_time = None

        show_fps = False
        fps_time = time.time()
        fps_counter = 0
        current_fps = 0

        print("\nEntering main detection mode. Press 'q' to quit, 'r' to reset message, 'f' to toggle FPS, 'c' to recalibrate.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_time = time.time()

            hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lowerLimit, upperLimit = get_limits(color=[0,255,0])
            mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
            mask_ = Image.fromarray(mask)
            bbox = mask_.getbbox()

            current_time = time.time()
            light_detected = bbox is not None

            # Debounce
            detection_history.append(light_detected)
            if len(detection_history) > DEBOUNCE_FRAMES:
                detection_history.pop(0)
            debounced_detection = all(detection_history) if len(detection_history) == DEBOUNCE_FRAMES else light_detected

            # Draw bounding box
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                green_region = hsvImage[y1:y2, x1:x2]
                avg_value = np.mean(green_region[:, :, 2]) if green_region.size > 0 else 0
                cv2.putText(frame, f"Brightness: {int(avg_value)}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Track timings
            if debounced_detection:
                if not light_visible:
                    light_visible = True
                    visibility_start_time = current_time
                last_visible_time = current_time
            else:
                if light_visible:
                    duration = current_time - visibility_start_time
                    # classify
                    if duration < dot_dash_cutoff:
                        current_morse += "."
                        print(f"DOT (.) - Duration: {duration:.2f}s")
                    else:
                        current_morse += "-"
                        print(f"DASH (-) - Duration: {duration:.2f}s")
                    light_visible = False
                    visibility_start_time = None

            # Check gaps to decode letters & words
            if last_visible_time is not None and not light_visible:
                time_since_last = current_time - last_visible_time
                if time_since_last > gap_threshold and current_morse:
                    # letter boundary
                    if current_morse in MORSE_CODE:
                        letter = MORSE_CODE[current_morse]
                        decoded_message += letter
                        print(f"\n'{current_morse}' -> {letter}")
                        print(f"Message so far: {decoded_message}\n")
                    else:
                        print(f"\nUnknown morse: {current_morse}")
                        decoded_message += "?"
                    current_morse = ""
                if time_since_last > word_gap_threshold:
                    if decoded_message and decoded_message[-1] != " ":
                        decoded_message += " "
                        print("--- WORD BREAK ---\n")

            # Display info on frame
            cv2.putText(frame, f"Current: {current_morse}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Message: {decoded_message}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if show_fps:
                cv2.putText(frame, f"FPS: {current_fps}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            if light_visible:
                cv2.circle(frame, (frame.shape[1]-30, 30), 15, (0,255,0), -1)
            else:
                cv2.circle(frame, (frame.shape[1]-30, 30), 15, (0,0,255), -1)

            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([frame, mask_colored])
            cv2.imshow('Morse Code Green LED Detector', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                current_morse = ""
                decoded_message = ""
                last_visible_time = None
                detection_history = []
                print("\n=== MESSAGE RESET ===\n")
            elif key == ord('f'):
                show_fps = not show_fps
                print(f"FPS display: {'ON' if show_fps else 'OFF'}")
            elif key == ord('c'):
                print("\n=== FORCE RECALIBRATION (word = SOS) ===\n")
                new_thresh = calibrate(cap, DEFAULT_CALIB_WORD, max_iterations=12, verbose=True)
                save_calibration(new_thresh)
                dot_dash_cutoff = new_thresh['dot_dash_cutoff']
                gap_threshold = new_thresh['gap']
                word_gap_threshold = new_thresh['word_gap']
                print("Recalibration done.")

    except KeyboardInterrupt as e:
        print("Interrupted:", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n=== Final Message ===")
        try:
            print(decoded_message)
        except:
            print("(no message)")


if _name_ == "_main_":
    run_main_detector()
