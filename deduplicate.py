import os
import hashlib
from music21 import converter, note, chord
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def get_pitch_sequence_worker(file_path):
    """
    Worker function to extract pitch sequence from a single file.
    Returns (file_path, file_size, hash_string) or None on failure.
    """
    try:
        # 'forceSource=True' ensures we don't pick up cached pickles which might be stale
        score = converter.parse(file_path, format='musicxml', forceSource=True)
        notes = []
        # Flattening is expensive, but necessary to get linear sequence across parts
        for element in score.flatten().notes:
            if isinstance(element, note.Note):
                notes.append(element.pitch.pitchClass)
            elif isinstance(element, chord.Chord):
                pcs = sorted([p.pitchClass for p in element.pitches])
                notes.extend(pcs)
            
            if len(notes) >= 20:
                break
        
        if not notes:
            return None

        seq = tuple(notes[:20])
        h = hashlib.md5(str(seq).encode()).hexdigest()
        size = os.path.getsize(file_path)
        return (file_path, size, h)
        
    except Exception:
        # Fail silently for speed/cleanliness in this worker
        return None

def deduplicate(mxl_dir):
    files = []
    for root, _, filenames in os.walk(mxl_dir):
        for f in filenames:
            if f.endswith('.mxl'):
                files.append(os.path.join(root, f))
    
    print(f"Found {len(files)} files. Starting parallel processing on {cpu_count()} cores...")
    
    hashes = {} # hash -> (file_path, file_size)
    results = []

    # Use multiprocessing to parse files
    with Pool(processes=cpu_count()) as pool:
        # tqdm wrapper around the iterator
        for result in tqdm(pool.imap_unordered(get_pitch_sequence_worker, files), total=len(files)):
            if result:
                results.append(result)

    print("Analysis complete. Resolving duplicates...")
    
    duplicates_removed = 0
    
    for file_path, size, h in results:
        if h in hashes:
            prev_path, prev_size = hashes[h]
            if size > prev_size:
                # New file is "better" (larger), delete old one
                try:
                    if os.path.exists(prev_path):
                        os.remove(prev_path)
                    hashes[h] = (file_path, size)
                    duplicates_removed += 1
                except OSError:
                    pass
            else:
                # Old file is better, delete new one
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    duplicates_removed += 1
                except OSError:
                    pass
        else:
            hashes[h] = (file_path, size)
            
    print(f"Deduplication complete. Removed {duplicates_removed} files.")

if __name__ == "__main__":
    deduplicate('mxl')
