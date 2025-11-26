#!/usr/bin/env python3
"""
Add a new trained model to the NoiseNCA demo.

Usage:
    python add_model.py --weights path/to/weights.pt --texture path/to/texture.jpg

This script:
1. Creates a directory in trained_models/Noise-NCA/ if needed and moves weights there
2. Converts the .pt file to JSON format in data/models/
3. Adds the texture name to data/metadata.json
"""

import os
import json
import argparse
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Add a new model to the NoiseNCA demo')
    parser.add_argument('--weights', type=str, required=True, help='Path to .pt weights file')
    parser.add_argument('--texture', type=str, required=True, help='Path to texture image (.jpg or .jpeg)')
    
    args = parser.parse_args()
    
    # Extract the base name from the texture path (e.g., "thick_vertical_stripes" from "data/images/texture/thick_vertical_stripes.jpg")
    texture_basename = os.path.basename(args.texture)
    texture_name = os.path.splitext(texture_basename)[0]
    
    print(f"Adding model for texture: {texture_name}")
    
    # 1. Create directory in trained_models/Noise-NCA/ and move weights
    model_dir = f"trained_models/Noise-NCA/{texture_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    weights_dest = os.path.join(model_dir, "weights.pt")
    if args.weights != weights_dest:
        shutil.copy(args.weights, weights_dest)
        print(f"Copied weights to: {weights_dest}")
    else:
        print(f"Weights already at: {weights_dest}")
    
    # 2. Convert .pt to JSON
    json_output = f"data/models/{texture_name}.json"
    print(f"Converting to JSON: {json_output}")
    
    result = subprocess.run(
        ["python", "convert_pt_to_json.py", weights_dest, json_output],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error converting to JSON: {result.stderr}")
        return 1
    else:
        print(f"Successfully created: {json_output}")
    
    # 3. Add texture name to metadata.json
    metadata_path = "data/metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if texture_name not in metadata['texture_names']:
        metadata['texture_names'].append(texture_name)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Added '{texture_name}' to {metadata_path}")
    else:
        print(f"'{texture_name}' already in {metadata_path}")
    
    print(f"\nDone! Model '{texture_name}' is ready for the demo.")
    return 0


if __name__ == '__main__':
    exit(main())

