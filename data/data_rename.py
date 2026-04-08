import os
import sys

def batch_rename(directory):
    # get directory name
    dirname = os.path.basename(os.path.abspath(directory))
    
    # get all files (excluding directories) in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # rename by number
    for i, filename in enumerate(files, 1):
        # extract extension (keep original extension)
        name, ext = os.path.splitext(filename)
        new_name = f"{dirname}_{i:03d}{ext}"  # use 3-digit number (e.g., 001, 002)
        
        # build full path
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        # rename
        os.rename(old_path, new_path)
        print(f"Renamed file: {filename} → {new_name}")
    
    print(f"Completed renaming, processed {len(files)} files.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_rename.py /path/to/directory")
        sys.exit(1)
    
    directory = sys.argv[1]
    batch_rename(directory)