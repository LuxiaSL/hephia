"""
clear_memory_db.py

A utility script to safely clear all data from the memory database
while preserving the schema structure.
"""

import os
import asyncio
import sqlite3
from pathlib import Path

# Default database path matching the memory system
DEFAULT_DB_PATH = "data/memory.db"

async def clear_memory_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Clears all data from memory database tables while preserving schema.
    
    Args:
        db_path: Path to the SQLite database file
    """
    try:
        # Verify database exists
        if not os.path.exists(db_path):
            print(f"Database not found at: {db_path}")
            return

        # Connect to database
        print(f"Connecting to database at: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        # Enable foreign key support
        cursor.execute("PRAGMA foreign_keys = OFF")
        
        try:
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")

            # Clear each table
            for (table_name,) in tables:
                if table_name != "sqlite_sequence":  # Skip internal SQLite table
                    print(f"Clearing table: {table_name}")
                    cursor.execute(f"DELETE FROM {table_name}")

            # Reset auto-increment counters if the table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
            if cursor.fetchone():
                cursor.execute("DELETE FROM sqlite_sequence")
            
            # Commit transaction
            conn.commit()
            print("Successfully cleared all data from database")

        except Exception as e:
            # Rollback on error
            conn.rollback()
            print(f"Error clearing database: {e}")
            raise

        finally:
            # Re-enable foreign keys
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Cleanup
            cursor.close()
            conn.close()

    except Exception as e:
        print(f"Failed to clear database: {e}")
        raise

def main():
    """Main entry point"""
    # Get database path from environment or use default
    db_path = os.getenv("MEMORY_DB_PATH", DEFAULT_DB_PATH)
    
    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing to clear memory database at: {db_path}")
    
    # Run the clear operation
    asyncio.run(clear_memory_database(db_path))
    
    print("Database clearing operation completed")

if __name__ == "__main__":
    main()