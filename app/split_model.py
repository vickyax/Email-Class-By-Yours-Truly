import os

def split_file(file_path, chunk_size=70 * 1024 * 1024): # 70 MB chunks
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        part_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            part_name = f"{file_path}.part{part_num}"
            with open(part_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            print(f"✅ Created {part_name} ({len(chunk)/1024/1024:.2f} MB)")
            part_num += 1
    
    print(f"🎉 Done! You can now delete '{file_path}' and push the '.part0' and '.part1' files to GitHub.")

# Run the split
split_file("quantized_model/quantized_bert.pth")