#!/usr/bin/env python3
"""
Three-window curses TUI with partial (line-by-line) redraw to reduce flicker,
plus dedicated logging, cross-platform compatibility, and support for
a "cognitive summary" panel.
"""

import curses
import threading
import time
from queue import Queue, Empty
from typing import Optional, List
import logging
from pathlib import Path
from datetime import datetime
import signal

# If you have a config module:
from config import Config

###############################################################################
# Logging Setup
###############################################################################
def setup_logging():
    log_dir = Path('data/logs/visualization')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = log_dir / f'vis-{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            # Important: keep the encoding here for Windows compatibility
            logging.FileHandler(log_file, encoding='utf-8'),
        ]
    )
    logger = logging.getLogger('visualization')
    # Ensure no console output
    logger.propagate = False
    return logger

logger = setup_logging()

###############################################################################
# Global Queues
###############################################################################
cognitive_queue = Queue()
state_queue = Queue()

###############################################################################
# WindowPanel Class (Partial Redraw)
###############################################################################
class WindowPanel:
    """
    A window panel that performs partial redraws to reduce flicker.
    Maintains old and new buffers, only re-drawing changed lines.
    """
    def __init__(self, win: curses.window, title: str = ""):
        self.win = win
        self.title = title

        # Store old and new text lines for partial re-draw
        self.old_buffer: List[str] = []
        self.new_buffer: List[str] = []

        # Initialize the panel
        self._init_panel()

    def _init_panel(self):
        """Initial draw of the panel's border/title."""
        try:
            self.win.box()
            if self.title:
                self.win.addstr(0, 2, f" {self.title} ")
            self.win.refresh()
        except curses.error as e:
            logger.error(f"Error initializing window panel '{self.title}': {e}")

    def set_content(self, lines: List[str]):
        """Set new content lines that will be partially redrawn on next render."""
        self.new_buffer = lines

    def render(self):
        """
        Compare new_buffer to old_buffer line by line, only redrawing changed lines.
        Draws a border + title each time (small overhead, but simpler to manage).
        """
        try:
            height, width = self.win.getmaxyx()
            usable_height = height - 2  # minus top/bottom border
            usable_width = width - 2    # minus left/right border

            # Redraw border/title
            self.win.box()
            if self.title:
                self.win.addstr(0, 2, f" {self.title} ")

            # We'll only display up to 'usable_height' lines:
            visible_new = self.new_buffer[:usable_height]
            visible_old = self.old_buffer[:usable_height]

            for row in range(usable_height):
                # Position in curses window (accounting for the border)
                wrow = row + 1
                wcol = 1

                new_line = ""
                old_line = ""
                if row < len(visible_new):
                    new_line = visible_new[row]
                if row < len(visible_old):
                    old_line = visible_old[row]

                # Redraw only if changed
                if new_line != old_line:
                    # Safely truncate the line to fit
                    display_line = new_line[:usable_width]
                    try:
                        self.win.move(wrow, wcol)
                        # Clear that line portion
                        self.win.clrtoeol()
                        self.win.addstr(wrow, wcol, display_line)
                    except curses.error:
                        # Typically harmless if the terminal is resizing mid-draw
                        pass

            # Refresh the panel
            self.win.refresh()

            # Save current state as old for next iteration
            self.old_buffer = self.new_buffer[:]

        except curses.error as e:
            logger.error(f"Error rendering panel '{self.title}': {e}", exc_info=True)

###############################################################################
# MonitorUI Class
###############################################################################
class MonitorUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr

        # We'll create WindowPanels for each region
        self.cognitive_panel: Optional[WindowPanel] = None
        self.state_panel: Optional[WindowPanel] = None
        self.summary_panel: Optional[WindowPanel] = None

        self.running = True
        self.last_size = None
        
        # For rate-limiting and deduplicating messages
        self.last_cognitive_msg = None
        self.last_state_msg = None
        self.last_full_summary_msg = None
        self._last_state_update = 0

        logger.info("Initializing Monitor UI")
        try:
            self.setup_windows()
            logger.info("Monitor UI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Monitor UI: {e}", exc_info=True)
            raise

    def setup_windows(self):
        """Create or recreate the curses windows and corresponding WindowPanels."""
        try:
            self.stdscr.clear()
            self.stdscr.refresh()
            
            # Get total screen size
            total_height, total_width = self.stdscr.getmaxyx()

            # Split screen: left half for cognitive, right half for state & summary
            left_width = total_width // 2
            right_width = total_width - left_width

            # Left panel: full height for cognitive
            cognitive_win = curses.newwin(total_height, left_width, 0, 0)
            
            # Right side: top half for state, bottom half for summary
            right_top_height = total_height // 2
            right_bottom_height = total_height - right_top_height

            state_win = curses.newwin(right_top_height, right_width, 0, left_width)
            summary_win = curses.newwin(right_bottom_height, right_width, right_top_height, left_width)

            # Build WindowPanel objects
            self.cognitive_panel = WindowPanel(cognitive_win, "Cognitive Processing")
            self.state_panel = WindowPanel(state_win, "System State")
            self.summary_panel = WindowPanel(summary_win, "Cognitive Summary")

            # Store last known size
            self.last_size = (total_height, total_width)
        except Exception as e:
            logger.error(f"Error setting up windows: {e}", exc_info=True)
            raise

    def run(self):
        """Main monitoring loop."""
        logger.info("Starting monitor loop")

        while self.running:
            try:
                # Check if terminal size changed
                if self.check_resize():
                    self.setup_windows()

                # Process all incoming messages
                self.process_cognitive_queue()
                self.process_state_queue()

                # Render each panel
                if self.cognitive_panel:
                    self.cognitive_panel.render()
                if self.state_panel:
                    self.state_panel.render()
                if self.summary_panel:
                    self.summary_panel.render()

                # Poll for user input (non-blocking)
                try:
                    ch = self.stdscr.getch()
                    if ch in (ord('q'), 3):  # 'q' or Ctrl-C
                        self.running = False
                except curses.error:
                    # Non-fatal if resizing
                    pass

                time.sleep(0.25)

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)
                try:
                    self.setup_windows()
                except:
                    logger.critical("Failed to recover from error", exc_info=True)
                    break

        logger.info("Monitor loop ended")

    def check_resize(self) -> bool:
        """Check if terminal has been resized."""
        try:
            current_size = self.stdscr.getmaxyx()
            if current_size != self.last_size:
                self.last_size = current_size
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking terminal size: {e}", exc_info=True)
            return False

    def process_cognitive_queue(self):
        """Dequeue and handle all cognitive events."""
        while True:
            try:
                event = cognitive_queue.get_nowait()
            except Empty:
                break
            self.update_cognitive_panel(event)

    def process_state_queue(self):
        """Dequeue and handle all state events."""
        while True:
            try:
                event = state_queue.get_nowait()
            except Empty:
                break
            self.update_state_panel(event)

    ###########################################################################
    # Panel Update Methods
    ###########################################################################
    def update_cognitive_panel(self, event):
        """
        Show truncated summary + recent messages in the Cognitive (left) panel,
        full un-truncated summary in the Summary (bottom-right) panel.
        """
        try:
            # Extract the main processed state
            full_summary = event.data.get('processed_state', 'No summary available')
            truncated_summary = (
                full_summary[:100] + "..." if len(full_summary) > 100 else full_summary
            )

            # Build the content for the cognitive window
            messages = event.data.get('raw_state', [])[-2:]
            formatted_messages = []
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    display_name = "EXO-PROCESSOR"
                    content = content[:500] + "..." if len(content) > 500 else content
                else:
                    # Use your config for the model name
                    display_name = Config.get_cognitive_model()
                    content = content[:5000] + "..." if len(content) > 5000 else content

                formatted_messages.append(f"  {display_name}: {content}")

            cognitive_message = (
                "Cognitive State:\n" +
                "Recent Messages:\n" +
                "\n".join(formatted_messages)
            )

            # Update left panel if changed
            if cognitive_message != self.last_cognitive_msg:
                self.last_cognitive_msg = cognitive_message
                # Convert to lines, wrapping if needed
                lines = self._split_and_wrap(cognitive_message, self.cognitive_panel.win)
                self.cognitive_panel.set_content(lines)

            # Update summary panel with full untruncated summary
            if full_summary != self.last_full_summary_msg:
                self.last_full_summary_msg = full_summary
                summary_lines = self._split_and_wrap(full_summary, self.summary_panel.win)
                self.summary_panel.set_content(summary_lines)

        except Exception as e:
            logger.error(f"Error updating cognitive panel: {e}", exc_info=True)

    def update_state_panel(self, event):
        """Show system state info in the top-right panel."""
        try:
            needs_data = event.data.get('context', {}).get('needs', {})
            needs_satisfaction = []
            for need, details in needs_data.items():
                if isinstance(details, dict) and 'satisfaction' in details:
                    pct = details['satisfaction'] * 100
                    needs_satisfaction.append(f"{need}: {pct:.2f}%")

            mood_data = event.data.get('context', {}).get('mood', {})
            mood_name = mood_data.get('name', 'unknown')
            valence = mood_data.get('valence', 0)
            arousal = mood_data.get('arousal', 0)
            behavior = event.data.get('context', {}).get('behavior', {}).get('name', 'none')
            emotional_state = event.data.get('context', {}).get('emotional_state', 'neutral')

            state_message = (
                "System State:\n"
                f"Mood: {mood_name} (v:{valence:.2f}, a:{arousal:.2f})\n"
                f"Behavior: {behavior}\n"
                f"Needs Satisfaction: {', '.join(needs_satisfaction)}\n"
                f"Emotional State: {emotional_state}"
            )

            current_time = time.time()
            # Rate-limit: only update if changed and at least 0.1s have passed
            if state_message != self.last_state_msg and current_time - self._last_state_update > 0.1:
                self.last_state_msg = state_message
                self._last_state_update = current_time

                lines = self._split_and_wrap(state_message, self.state_panel.win)
                self.state_panel.set_content(lines)

        except Exception as e:
            logger.error(f"Error updating state panel: {e}", exc_info=True)

    ###########################################################################
    # Utility Methods
    ###########################################################################
    def _split_and_wrap(self, text: str, win: curses.window) -> List[str]:
        """
        Splits text into lines and wraps according to window width,
        preventing horizontal overflow. 
        """
        lines_out = []
        try:
            # Get usable width (minus borders)
            height, width = win.getmaxyx()
            usable_width = max(1, width - 2)  # account for box borders

            paragraphs = text.split('\n')
            for paragraph in paragraphs:
                # Normalize spacing
                paragraph = ' '.join(paragraph.split())
                words = paragraph.split(' ')

                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 <= usable_width:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        lines_out.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)

                if current_line:
                    lines_out.append(' '.join(current_line))
                # Separate paragraphs
                lines_out.append("")
        except curses.error:
            # If window can't get size, fallback
            lines_out = text.split('\n')

        return lines_out

###############################################################################
# Main Entry Point
###############################################################################
def start_monitor():
    """Initialize and start the monitor via curses wrapper."""
    logger.info("Starting monitoring system")

    # Optionally: handle SIGWINCH for immediate resize detection
    def handle_resize(signum, frame):
        # Tells curses to resize its internal structures
        curses.resize_term(0, 0)

    # Only register SIGWINCH if it exists on this platform
    if hasattr(signal, 'SIGWINCH'):
        signal.signal(signal.SIGWINCH, handle_resize)

    def run_wrapped(stdscr):
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.curs_set(0)  # Hide cursor
            stdscr.nodelay(True)  # Non-blocking input

            ui = MonitorUI(stdscr)
            ui.run()
        except Exception as e:
            logger.critical(f"Fatal error in monitor: {e}", exc_info=True)
            raise
        finally:
            logger.info("Monitor system shutdown")

    try:
        curses.wrapper(run_wrapped)
    except Exception as e:
        logger.critical(f"Failed to start monitor: {e}", exc_info=True)
        raise

###############################################################################
# External Event Handlers
###############################################################################
def handle_cognitive_event(event_data):
    """Queue a cognitive event for display in the monitor."""
    try:
        cognitive_queue.put(event_data)
    except Exception as e:
        logger.error(f"Error handling cognitive event: {e}", exc_info=True)

def handle_state_event(event_data):
    """Queue a state event for display in the monitor."""
    try:
        state_queue.put(event_data)
    except Exception as e:
        logger.error(f"Error handling state event: {e}", exc_info=True)
