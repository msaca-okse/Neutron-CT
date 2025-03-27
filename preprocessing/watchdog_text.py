import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Triggered when a new file is created."""
        if not event.is_directory:
            print(f"File created: {event.src_path}")

    def on_modified(self, event):
        """Triggered when a file is modified."""
        if not event.is_directory:
            print(f"File modified: {event.src_path}")

    def on_moved(self, event):
        """Triggered when a file is moved."""
        if not event.is_directory:
            print(f"File moved: {event.src_path}")

if __name__ == "__main__":

    FOLDER_TO_WATCH = '/work3/msaca/sliceA_DA_cache'  # Ensure this is the correct path to the folder you want to watch
    event_handler = FileHandler()  # Initialize the event handler

    # Create and start the observer
    observer = Observer()
    observer.schedule(event_handler, FOLDER_TO_WATCH, recursive=False)
    observer.start()

    observer._use_polling = True

    print(f"Watching directory: {FOLDER_TO_WATCH}")

    try:
        while True:  # Keep the script running to listen for events
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        observer.join()
        print("Stopped watching.")