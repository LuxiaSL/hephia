# visualization.py

import curses
import threading
import time
from queue import Queue, Empty
from typing import Tuple, Optional
import os
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
def setup_logging():
    """Configure logging for visualization system"""
    # Create logs directory if it doesn't exist
    log_dir = Path('data/logs/visualization')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = log_dir / f'vis-{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    return logging.getLogger('visualization')

logger = setup_logging()

# Global queues
cognitive_queue = Queue()
state_queue = Queue()
command_queue = Queue()

class TerminalUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.cognitive_win: Optional[curses.window] = None
        self.state_win: Optional[curses.window] = None
        self.command_win: Optional[curses.window] = None
        self.user_input = ""
        self.running = True
        self.last_size = None
        
        logger.info("Initializing Terminal UI")
        try:
            self.setup_windows()
            logger.info("Terminal UI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Terminal UI: {e}", exc_info=True)
            raise

    def check_resize(self) -> bool:
        """Check if terminal has been resized"""
        try:
            current_size = self.stdscr.getmaxyx()
            if current_size != self.last_size:
                logger.debug(f"Terminal resized from {self.last_size} to {current_size}")
                self.last_size = current_size
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking terminal size: {e}", exc_info=True)
            return False

    def get_window_dimensions(self) -> Tuple[int, int, int, int, int, int]:
        """Calculate window dimensions based on terminal size"""
        try:
            height, width = self.stdscr.getmaxyx()
            logger.debug(f"Terminal dimensions: {width}x{height}")
            
            top_height = max(int(height * 0.75), 5)
            bot_height = max(height - top_height, 3)
            left_width = max(int(width * 0.5), 20)
            right_width = max(width - left_width, 20)
            
            dimensions = (top_height, bot_height, left_width, right_width, height, width)
            logger.debug(f"Calculated dimensions: {dimensions}")
            return dimensions
        except Exception as e:
            logger.error(f"Error calculating window dimensions: {e}", exc_info=True)
            # Return safe default dimensions
            return (20, 5, 40, 40, 25, 80)

    def setup_windows(self):
        """Create or recreate all windows with current terminal dimensions"""
        try:
            logger.debug("Setting up windows")
            self.stdscr.clear()
            self.stdscr.refresh()
            
            dimensions = self.get_window_dimensions()
            top_height, bot_height, left_width, right_width, height, width = dimensions
            
            # Create new windows
            self.cognitive_win = curses.newwin(top_height, left_width, 0, 0)
            self.state_win = curses.newwin(top_height, right_width, 0, left_width)
            self.command_win = curses.newwin(bot_height, width, top_height, 0)
            
            # Setup window properties
            for win in [self.cognitive_win, self.state_win, self.command_win]:
                win.scrollok(True)
                win.idlok(True)
            
            # Draw borders and titles
            self._safe_draw_borders()
            self.last_size = self.stdscr.getmaxyx()
            logger.info("Windows setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up windows: {e}", exc_info=True)
            raise

    def _safe_draw_borders(self):
        """Safely draw borders and titles for all windows"""
        try:
            for win, title in [
                (self.cognitive_win, " Cognitive Processing "),
                (self.state_win, " System State "),
                (self.command_win, " Command Input ")
            ]:
                win.border()
                win.addstr(0, 2, title)
                win.refresh()
            
            self.update_command_panel(f"> {self.user_input}")
        except curses.error as e:
            logger.error(f"Error drawing borders: {e}")

    def update_cognitive_panel(self, message: str):
        """Update cognitive panel with new message"""
        try:
            logger.debug(f"Updating cognitive panel: {message[:50]}...")
            height = self.cognitive_win.getmaxyx()[0]
            width = self.cognitive_win.getmaxyx()[1] - 2
            
            wrapped_lines = [message[i:i+width] for i in range(0, len(message), width)]
            
            for line in wrapped_lines:
                self.cognitive_win.scroll(1)
                self.cognitive_win.addstr(height-2, 1, line[:width])
            
            self._safe_draw_borders()
            
        except Exception as e:
            logger.error(f"Error updating cognitive panel: {e}", exc_info=True)

    def update_state_panel(self, message: str):
        """Update state panel with new message"""
        try:
            logger.debug(f"Updating state panel: {message[:50]}...")
            height = self.state_win.getmaxyx()[0]
            width = self.state_win.getmaxyx()[1] - 2
            
            wrapped_lines = [message[i:i+width] for i in range(0, len(message), width)]
            
            for line in wrapped_lines:
                self.state_win.scroll(1)
                self.state_win.addstr(height-2, 1, line[:width])
            
            self._safe_draw_borders()
            
        except Exception as e:
            logger.error(f"Error updating state panel: {e}", exc_info=True)

    def update_command_panel(self, prompt: str):
        """Update command panel with current input"""
        try:
            logger.debug(f"Updating command panel: {prompt[:50]}...")
            self.command_win.clear()
            self.command_win.border()
            self.command_win.addstr(0, 2, " Command Input ")
            self.command_win.addstr(1, 1, prompt[:self.command_win.getmaxyx()[1]-3])
            self.command_win.refresh()
        except Exception as e:
            logger.error(f"Error updating command panel: {e}", exc_info=True)

    def run(self):
        """Main UI loop"""
        logger.info("Starting main UI loop")
        while self.running:
            try:
                # Check for resize
                if self.check_resize():
                    logger.info("Handling terminal resize")
                    self.setup_windows()
                
                # Process queued messages
                self._process_message_queues()
                
                # Handle user input
                self._handle_user_input()
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                try:
                    self.setup_windows()
                except:
                    logger.critical("Failed to recover from error", exc_info=True)
                    self.running = False
                    break

        logger.info("UI loop terminated")

    def _process_message_queues(self):
        """Process all pending messages from queues"""
        try:
            # Process cognitive updates
            while True:
                try:
                    msg = cognitive_queue.get_nowait()
                    self.update_cognitive_panel(msg)
                except Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing cognitive message: {e}", exc_info=True)

            # Process state updates
            while True:
                try:
                    msg = state_queue.get_nowait()
                    self.update_state_panel(msg)
                except Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing state message: {e}", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Error in message queue processing: {e}", exc_info=True)

    def _handle_user_input(self):
        """Handle user input processing"""
        try:
            ch = self.stdscr.getch()
            if ch == -1:
                return  # No input
            elif ch in (curses.KEY_ENTER, 10, 13):
                if self.user_input.strip():
                    logger.debug(f"Processing command: {self.user_input}")
                    command_queue.put(self.user_input)
                    self.update_cognitive_panel(f"User: {self.user_input}")
                self.user_input = ""
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                self.user_input = self.user_input[:-1]
            elif ch == 3:  # ctrl+c
                logger.info("Received Ctrl+C, shutting down")
                self.running = False
            elif 32 <= ch <= 126:  # Printable characters
                max_input = self.command_win.getmaxyx()[1] - 4
                if len(self.user_input) < max_input:
                    self.user_input += chr(ch)

            self.update_command_panel(f"> {self.user_input}")
            
        except Exception as e:
            logger.error(f"Error handling user input: {e}", exc_info=True)

def start_visualization():
    """Initialize and start the UI"""
    logger.info("Starting visualization system")
    
    def run_wrapped(stdscr):
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.curs_set(0)
            stdscr.nodelay(True)
            
            ui = TerminalUI(stdscr)
            ui.run()
            
        except Exception as e:
            logger.critical(f"Fatal error in visualization: {e}", exc_info=True)
            raise
        finally:
            logger.info("Visualization system shutdown")
    
    try:
        curses.wrapper(run_wrapped)
    except Exception as e:
        logger.critical(f"Failed to start visualization: {e}", exc_info=True)
        raise

def handle_cognitive_event(event_data: str):
    """Queue cognitive event for display"""
    try:
        logger.debug(f"Received cognitive event: {event_data[:50]}...")
        cognitive_queue.put(event_data)
    except Exception as e:
        logger.error(f"Error handling cognitive event: {e}", exc_info=True)

def handle_state_event(event_data: str):
    """Queue state event for display"""
    try:
        logger.debug(f"Received state event: {event_data[:50]}...")
        state_queue.put(event_data)
    except Exception as e:
        logger.error(f"Error handling state event: {e}", exc_info=True)

def poll_commands() -> Optional[str]:
    """Poll for user commands"""
    try:
        cmd = command_queue.get_nowait()
        logger.debug(f"Retrieved command: {cmd}")
        return cmd
    except Empty:
        return None
    except Exception as e:
        logger.error(f"Error polling commands: {e}", exc_info=True)
        return None