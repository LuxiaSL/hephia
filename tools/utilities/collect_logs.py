import re
import json
import os
import glob
from datetime import datetime

def group_log_entries(log_file):
    """
    Reads the log file and groups lines into complete log entries.
    Each new entry is assumed to begin with a timestamp pattern.
    """
    entry_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?) \| ')
    entries = []
    current_lines = []
    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if entry_pattern.match(line):
                if current_lines:
                    entries.append(''.join(current_lines))
                    current_lines = []
            current_lines.append(line)
        if current_lines:
            entries.append(''.join(current_lines))
    return entries

def parse_event(entry, source_file):
    """
    Parses a complete log entry for LLM content events.
    We check for:
      - "Raw API response"
      - "One-turn LLM call succeeded"
      - "LLM EXCHANGE"
      - "Terminal Response:" (from BrainLogger.debug)
      - "State Context:" (from BrainLogger.debug)
    """
    # Filter out entries that don't mention our key phrases.
    if not any(phrase in entry for phrase in (
        "Raw API response",
        "One-turn LLM call succeeded",
        "LLM EXCHANGE",
        "Terminal Response:",
        "State Context:"
    )):
        return None

    # Extract timestamp from the beginning of the entry.
    timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?)', entry)
    if not timestamp_match:
        return None
    timestamp_str = timestamp_match.group(1)
    try:
        if ',' in timestamp_str:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        else:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        return None

    event = {
        'timestamp': timestamp.isoformat(),
        'source_file': source_file,
        'raw': entry
    }

    # --- API RESPONSE Events ---
    if "Raw API response" in entry:
        title_match = re.search(r'Raw API response \((.*?)\):', entry)
        api_title = title_match.group(1) if title_match else "Unknown"
        try:
            split_token = f'Raw API response ({api_title}):'
            json_str = entry.split(split_token, 1)[1].strip()
            data = json.loads(json_str)
            content_texts = [c.get('text', '') for c in data.get('content', []) if c.get('type') == 'text']
            event.update({
                'type': 'API_RESPONSE',
                'api_title': api_title,
                'model': data.get('model', ''),
                'content': content_texts,
                'usage': data.get('usage', {})
            })
        except Exception as e:
            event['error'] = f"Error parsing API response: {e}"

    # --- One-Turn LLM Call Succeeded (Summary) ---
    elif "One-turn LLM call succeeded" in entry:
        try:
            parts = entry.split('One-turn LLM call succeeded for', 1)
            summary_part = parts[1]
            summary_split = summary_part.split(':', 1)
            call_identifier = summary_split[0].strip()
            summary_text = summary_split[1].strip()
            event.update({
                'type': 'SUMMARY',
                'call_identifier': call_identifier,
                'content': summary_text,
                'components': re.findall(r'\b[A-Z][a-zA-Z]+\b', summary_text)
            })
        except Exception as e:
            event['error'] = f"Error parsing SUMMARY event: {e}"

    # --- LLM Exchange ---
    elif "LLM EXCHANGE" in entry:
        try:
            context_match = re.search(r'Context \(last 5 messages\):\s*(\{.*?\})', entry, re.DOTALL)
            response_match = re.search(r'Response:\s*(.*)', entry, re.DOTALL)
            context_data = json.loads(context_match.group(1)) if context_match else None
            response_text = response_match.group(1).strip() if response_match else ""
            event.update({
                'type': 'LLM_EXCHANGE',
                'context': context_data,
                'response': response_text
            })
        except Exception as e:
            event['error'] = f"Error parsing LLM_EXCHANGE event: {e}"

    # --- Terminal Response (from BrainLogger.debug) ---
    elif "Terminal Response:" in entry:
        try:
            match = re.search(r'Terminal Response:\s*(.*)', entry, re.DOTALL)
            terminal_response = match.group(1).strip() if match else ""
            event.update({
                'type': 'TERMINAL_RESPONSE',
                'terminal_response': terminal_response
            })
        except Exception as e:
            event['error'] = f"Error parsing TERMINAL_RESPONSE event: {e}"

    # --- State Context (from BrainLogger.debug) ---
    elif "State Context:" in entry:
        try:
            match = re.search(r'State Context:\s*(.*)', entry, re.DOTALL)
            state_context = match.group(1).strip() if match else ""
            event.update({
                'type': 'STATE_CONTEXT',
                'state_context': state_context
            })
        except Exception as e:
            event['error'] = f"Error parsing STATE_CONTEXT event: {e}"

    return event

def parse_logs_from_file(log_file):
    entries = group_log_entries(log_file)
    events = []
    for entry in entries:
        event = parse_event(entry, os.path.basename(log_file))
        if event:
            events.append(event)
    return events

def combine_related_events(events, threshold_seconds=1.0):
    """
    Combines consecutive TERMINAL_RESPONSE and STATE_CONTEXT events if they occur
    within `threshold_seconds` of each other. The assumption is that a terminal response
    is immediately followed by a state context update.
    """
    if not events:
        return events

    combined = []
    i = 0
    while i < len(events):
        event = events[i]
        # Check if the event is one that can be combined.
        if event.get('type') in ('TERMINAL_RESPONSE', 'STATE_CONTEXT'):
            composite = event.copy()
            t_current = datetime.fromisoformat(event['timestamp'])
            j = i + 1
            while j < len(events) and events[j].get('type') in ('TERMINAL_RESPONSE', 'STATE_CONTEXT'):
                t_next = datetime.fromisoformat(events[j]['timestamp'])
                delta = (t_next - t_current).total_seconds()
                if delta <= threshold_seconds:
                    # Merge fields: if next event is STATE_CONTEXT, add/update it.
                    if events[j]['type'] == 'STATE_CONTEXT':
                        composite['state_context'] = events[j].get('state_context')
                    elif events[j]['type'] == 'TERMINAL_RESPONSE':
                        if 'terminal_response' in composite:
                            composite['terminal_response'] += "\n" + events[j].get('terminal_response', '')
                        else:
                            composite['terminal_response'] = events[j].get('terminal_response', '')
                    composite['timestamp'] = events[j]['timestamp']
                    j += 1
                else:
                    break
            combined.append(composite)
            i = j
        else:
            combined.append(event)
            i += 1
    return combined

def generate_timeline_document(events, output_path):
    """
    Generate a Markdown timeline document from the list of LLM content events.
    """
    lines = []
    lines.append("# Cognitive Events Timeline\n")
    lines.append("This document summarizes LLM-generated events including API responses, one-turn summaries, "
                 "LLM exchanges, terminal responses, and state context updates.\n")
    
    # First, sort events by timestamp.
    sorted_events = sorted(events, key=lambda ev: ev['timestamp'])
    # Then combine related events.
    combined_events = combine_related_events(sorted_events, threshold_seconds=1.0)
    
    for event in combined_events:
        lines.append(f"## {event['timestamp']} ({event['source_file']})")
        lines.append(f"**Type:** {event.get('type', 'Unknown')}")
        
        if event.get('type') == 'API_RESPONSE':
            lines.append(f"**API Title:** {event.get('api_title', 'N/A')}")
            lines.append(f"**Model:** {event.get('model', 'N/A')}")
            if event.get('content'):
                lines.append("**Content:**")
                for text in event.get('content', []):
                    lines.append(f"> {text}")
            if event.get('usage'):
                lines.append("**Usage:**")
                lines.append("```json")
                lines.append(json.dumps(event.get('usage', {}), indent=2))
                lines.append("```")
        
        elif event.get('type') == 'SUMMARY':
            lines.append(f"**Call Identifier:** {event.get('call_identifier', 'N/A')}")
            lines.append("**Summary Content:**")
            lines.append(f"> {event.get('content', '')}")
            if event.get('components'):
                lines.append(f"**Components:** {', '.join(event.get('components', []))}")
        
        elif event.get('type') == 'LLM_EXCHANGE':
            lines.append("**LLM Exchange Details:**")
            if event.get('context'):
                lines.append("**Context (last 5 messages):**")
                lines.append("```json")
                lines.append(json.dumps(event.get('context'), indent=2))
                lines.append("```")
            lines.append("**Response:**")
            lines.append(f"> {event.get('response', '')}")
        
        elif event.get('type') == 'TERMINAL_RESPONSE':
            lines.append("**Terminal Response:**")
            lines.append("```")
            lines.append(event.get('terminal_response', ''))
            lines.append("```")
        
        elif event.get('type') == 'STATE_CONTEXT':
            lines.append("**State Context:**")
            lines.append("```")
            lines.append(event.get('state_context', ''))
            lines.append("```")
        
        if event.get('error'):
            lines.append(f"**Error:** {event.get('error')}")
        
        lines.append("\n---\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Timeline document generated at {output_path}")

if __name__ == "__main__":
    # Collect all visualization log files.
    log_dir = "./data/logs/visualization/"
    out_dir = "./data/"
    log_files = glob.glob(os.path.join(log_dir, "vis-*.log"))
    if not log_files:
        print("No visualization log files found")
        exit(1)
    
    all_events = []
    for log_file in log_files:
        events = parse_logs_from_file(log_file)
        all_events.extend(events)
    
    # Write aggregated parsed data to a JSON file.
    output_json = os.path.join(out_dir, "cognitive_events.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_events, f, indent=2)
    print(f"Parsed {len(all_events)} events from {len(log_files)} files.")
    
    # Generate a Markdown timeline document.
    timeline_md = os.path.join(out_dir, "cognitive_timeline.md")
    generate_timeline_document(all_events, timeline_md)
