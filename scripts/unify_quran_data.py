import os
import json
import re

def unify_quran():
    base_path = "/Users/hamid/Desktop/KivyxProjects/Islam_Encyclo/src/shared/data/"
    files = [f"quranPart{i}.ts" for i in range(1, 6)]
    
    full_dataset = []
    
    for filename in files:
        full_path = os.path.join(base_path, filename)
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} not found.")
            continue
            
        print(f"Reading {filename}...")
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract content between [ and ]
            match = re.search(r'\[(.*)\]', content, re.DOTALL)
            if match:
                data_str = match.group(0)
                try:
                    # Clean up common JS/TS syntax that JSON doesn't like
                    # 1. Remove trailing commas before closing braces/brackets
                    data_str = re.sub(r',\s*\]', ']', data_str)
                    data_str = re.sub(r',\s*\}', '}', data_str)
                    
                    data = json.loads(data_str)
                    full_dataset.extend(data)
                    print(f"Successfully loaded {len(data)} verses from {filename}")
                except Exception as e:
                    print(f"Error parsing {filename}: {e}")
                    # Fallback to a slightly more aggressive cleaning if needed
                    # but for now let's see if this works.
            else:
                print(f"Could not find array in {filename}")
                
    output_path = "/Users/hamid/Desktop/KivyxProjects/quran-ai-backend/quran_full.json"
    print(f"Saving unified dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=4)
    print(f"Completed! Total verses: {len(full_dataset)}")

if __name__ == "__main__":
    unify_quran()
