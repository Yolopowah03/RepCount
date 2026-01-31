from ..config import CHUNK_SIZE  # type: ignore
import os

def stream_file(file_path: str, range_header: str | None, mediatype: str):
    file_size = os.path.getsize(file_path)
    
    #Si l'arxiu és petit (no hi ha Range), enviar-lo sencer
    if not range_header:
        with open(file_path, mode="rb") as file:
            yield from file
        return
    
    range_match = range_header.strip().lower().split('bytes=')[1].split('-')
    
    start = int(range_match[0])
    end = file_size - 1
    
    if len(range_match) > 1 and range_match[1]:
        end = int(range_match[1])
        
    length = end - start + 1
    
    with open(file_path, mode="rb") as file:
        file.seek(start)
        bytes_read = 0
        while bytes_read < length:
            read_size = min(CHUNK_SIZE, length - bytes_read)
            data = file.read(read_size)
            if not data:
                break
            yield data
            bytes_read += len(data)