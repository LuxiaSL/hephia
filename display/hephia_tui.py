# display/monitor_tui.py
import curses
import threading
import time
from queue import Queue, Empty
from typing import Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime
from config import Config

# Setup logging
def setup_logging():
    log_dir = Path('data/logs/visualization')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = log_dir / f'vis-{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('visualization')

logger = setup_logging()

# Global event queues
cognitive_queue = Queue()
state_queue = Queue()

class MonitorUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.cognitive_win: Optional[curses.window] = None
        self.state_win: Optional[curses.window] = None
        self.running = True
        self.last_size = None
        self.last_cognitive_msg = None  # Cache for cognitive messages
        
        logger.info("Initializing Monitor UI")
        try:
            self.setup_windows()
            logger.info("Monitor UI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Monitor UI: {e}", exc_info=True)
            raise

    def get_window_dimensions(self) -> Tuple[int, int, int, int]:
        """Calculate window dimensions"""
        height, width = self.stdscr.getmaxyx()
        
        # Split screen evenly horizontally
        left_width = width // 2
        right_width = width - left_width
        
        return (height, left_width, right_width, width)

    def setup_windows(self):
        """Create or recreate the monitoring windows"""
        try:
            self.stdscr.clear()
            self.stdscr.refresh()
            
            height, left_width, right_width, _ = self.get_window_dimensions()
            
            # Create windows
            self.cognitive_win = curses.newwin(height, left_width, 0, 0)
            self.state_win = curses.newwin(height, right_width, 0, left_width)
            
            # Enable scrolling for cognitive window
            self.cognitive_win.scrollok(True)
            
            # Draw borders and titles
            self._draw_borders()
            self.last_size = self.stdscr.getmaxyx()
            
        except Exception as e:
            logger.error(f"Error setting up windows: {e}", exc_info=True)
            raise

    def _draw_borders(self):
        """Draw window borders and titles"""
        try:
            self.cognitive_win.box()
            self.state_win.box()
            
            # Add titles
            self.cognitive_win.addstr(0, 2, " Cognitive Processing ")
            self.state_win.addstr(0, 2, " System State ")
            
            self.cognitive_win.refresh()
            self.state_win.refresh()
        except curses.error as e:
            logger.error(f"Error drawing borders: {e}")

    def update_cognitive_panel(self, event):
        try:
            # Format cognitive state message
            summary = event.data.get('processed_state', 'No summary available')
            # Truncate summary to first 100 chars
            summary = summary[:100] + "..." if len(summary) > 100 else summary
            
            # Get last 2 messages from raw state
            messages = event.data.get('raw_state', [])[-2:]
            
            formatted_messages = []
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role == 'user':
                    # For user messages, show as EXO-PROCESSOR with shortened content
                    display_name = "EXO-PROCESSOR"
                    content = content[:500] + "..." if len(content) > 500 else content
                else:
                    # For assistant messages, use cognitive model name
                    display_name = Config.get_cognitive_model()
                    content = content[:1000] + "..." if len(content) > 1000 else content
                
                formatted_messages.append(f"  {display_name}: {content}")

            message = (
                f"Cognitive State:\n"
                f"Summary: {summary}\n"
                f"Recent Messages:\n" + 
                "\n".join(formatted_messages)
            )
            
            if message != self.last_cognitive_msg:
                self.last_cognitive_msg = message
                height, width = self.cognitive_win.getmaxyx()
                usable_width = width - 4
                
                # Split into paragraphs first
                paragraphs = message.split('\n')
                
                for paragraph in paragraphs:
                    # Clean up extra spaces within each paragraph
                    paragraph = ' '.join(paragraph.split())
                    
                    # Word wrap each paragraph
                    words = paragraph.split(' ')
                    current_line = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 <= usable_width:
                            current_line.append(word)
                            current_length += len(word) + 1
                        else:
                            # Write current line
                            self.cognitive_win.scroll(1)
                            line_text = ' '.join(current_line)
                            self.cognitive_win.addstr(height-2, 2, line_text.ljust(usable_width))
                            current_line = [word]
                            current_length = len(word)
                    
                    # Write final line of paragraph
                    if current_line:
                        self.cognitive_win.scroll(1)
                        line_text = ' '.join(current_line)
                        self.cognitive_win.addstr(height-2, 2, line_text.ljust(usable_width))
                    
                    # Add an empty line between paragraphs
                    self.cognitive_win.scroll(1)
                    self.cognitive_win.addstr(height-2, 2, " " * usable_width)
                
                self._draw_borders()
        except Exception as e:
            logger.error(f"Error updating cognitive panel: {e}", exc_info=True)

    def update_state_panel(self, event):
        try:
            # Extract and format needs satisfaction percentages
            needs_data = event.data.get('context', {}).get('needs', {})
            needs_satisfaction = []
            for need, details in needs_data.items():
                if isinstance(details, dict) and 'satisfaction' in details:
                    satisfaction_pct = details['satisfaction'] * 100
                    needs_satisfaction.append(f"{need}: {satisfaction_pct:.2f}%")

            message = (
                f"System State:\n"
                f"Mood: {event.data.get('context', {}).get('mood', {}).get('name', 'unknown')} "
                f"(v:{event.data.get('context', {}).get('mood', {}).get('valence', 0):.2f}, "
                f"a:{event.data.get('context', {}).get('mood', {}).get('arousal', 0):.2f})\n"
                f"Behavior: {event.data.get('context', {}).get('behavior', {}).get('name', 'none')}\n"
                f"Needs Satisfaction: {', '.join(needs_satisfaction)}\n"
                f"Emotional State: {event.data.get('context', {}).get('emotional_state', 'neutral')}"
            )
            current_time = time.time()
            if not hasattr(self, '_last_state_update'):
                self._last_state_update = 0
                self._last_state_msg = None
                
            # Only update if message changed and at least 100ms passed
            if (message != self._last_state_msg and 
                current_time - self._last_state_update > 0.1):
                
                self._last_state_msg = message
                self._last_state_update = current_time
                
                # Clear window first
                height, width = self.state_win.getmaxyx()
                for y in range(1, height-1):
                    self.state_win.addstr(y, 1, " " * (width-2))
                
                usable_width = width - 4
                
                # Split into paragraphs first
                paragraphs = message.split('\n')
                
                current_y = 1
                for paragraph in paragraphs:
                    # Clean up extra spaces within each paragraph
                    paragraph = ' '.join(paragraph.split())
                    if 'value' in paragraph:
                        paragraph = paragraph.replace('"', '').replace('{', '').replace('}', '')
                    
                    # Word wrap each paragraph
                    words = paragraph.split(' ')
                    current_line = []
                    current_length = 0
                    
                    for word in words:
                        if current_length + len(word) + 1 <= usable_width:
                            current_line.append(word)
                            current_length += len(word) + 1
                        else:
                            if current_y < height-1:
                                line_text = ' '.join(current_line)
                                self.state_win.addstr(current_y, 2, line_text)
                                current_y += 1
                                current_line = [word]
                                current_length = len(word)
                    
                    # Write final line of paragraph
                    if current_line and current_y < height-1:
                        line_text = ' '.join(current_line)
                        self.state_win.addstr(current_y, 2, line_text)
                        current_y += 1
                
                self._draw_borders()
                
        except Exception as e:
            logger.error(f"Error updating state panel: {e}", exc_info=True)

    def check_resize(self) -> bool:
        """Check if terminal has been resized"""
        try:
            current_size = self.stdscr.getmaxyx()
            if current_size != self.last_size:
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking terminal size: {e}", exc_info=True)
            return False

    def run(self):
        """Main monitoring loop"""
        logger.info("Starting monitor loop")
        
        while self.running:
            try:
                # Handle resize if needed
                if self.check_resize():
                    self.setup_windows()
                
                # Process cognitive updates
                try:
                    while True:
                        msg = cognitive_queue.get_nowait()
                        self.update_cognitive_panel(msg)
                except Empty:
                    pass
                
                # Process state updates
                try:
                    while True:
                        msg = state_queue.get_nowait()
                        self.update_state_panel(msg)
                except Empty:
                    pass
                
                # Check for quit command (q or ctrl-c)
                try:
                    ch = self.stdscr.getch()
                    if ch in (ord('q'), 3):
                        self.running = False
                        break
                except curses.error:
                    pass
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)
                try:
                    self.setup_windows()
                except:
                    logger.critical("Failed to recover from error", exc_info=True)
                    break

def start_monitor():
    """Initialize and start the monitor"""
    logger.info("Starting monitoring system")
    
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

def handle_cognitive_event(event_data: str):
    """Queue cognitive event for display"""
    try:
        cognitive_queue.put(event_data)
    except Exception as e:
        logger.error(f"Error handling cognitive event: {e}", exc_info=True)

def handle_state_event(event_data: str):
    """Queue state event for display"""
    try:
        state_queue.put(event_data)
    except Exception as e:
        logger.error(f"Error handling state event: {e}", exc_info=True)