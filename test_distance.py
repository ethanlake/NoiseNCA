#!/usr/bin/env python3
"""
Test script to verify compute_pixel_distance behavior.
Creates r×r images where one is all black and the other has n white pixels,
then computes and plots the distance as a function of n.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Fix for matplotlib compatibility issue
try:
    import matplotlib.cbook
    if not hasattr(matplotlib.cbook, "_Stack"):
        class _Stack(list):
            def push(self, item):
                self.append(item)
                return item
            def pop(self):
                return super().pop() if self else None
            def current(self):
                return self[-1] if self else None
        matplotlib.cbook._Stack = _Stack
except:
    pass


def compute_pixel_distance(s1, s2):
    """
    Compute pixel-wise distance between two states using only RGB channels.
    
    Args:
        s1, s2: State tensors of shape [b, chn, h, w]
    
    Returns:
        Average pixel-wise distance (scalar) computed using only the first 3 channels (RGB)
    """
    # Compute L2 distance per pixel using only RGB channels (first 3 channels)
    diff = s1[:, :3, :, :] - s2[:, :3, :, :]  # Only use RGB channels
    pixel_distances = torch.norm(diff, dim=1)  # [b, h, w] - L2 norm over RGB channels
    return pixel_distances.mean().item()


def main():
    r = 20  # Image size: r×r
    total_pixels = r * r
    
    # Create state tensors: [batch=1, channels=3, height=r, width=r]
    # Black pixel in state space: [-0.5, -0.5, -0.5] (becomes [0, 0, 0] in RGB after to_rgb)
    # White pixel in state space: [0.5, 0.5, 0.5] (becomes [1, 1, 1] in RGB after to_rgb)
    
    # img1: all black
    s1 = torch.full((1, 3, r, r), -0.5)
    
    # Array to store distances
    n_values = []
    distances = []
    
    print(f"Testing distance calculation for r={r} (total pixels: {total_pixels})")
    print("Creating images with n white pixels and computing distances...")
    
    # Test n from 1 to r²
    for n in range(1, total_pixels + 1):
        # img2: all black except n white pixels
        s2 = torch.full((1, 3, r, r), -0.5)
        
        # Randomly select n pixels to make white
        # Flatten spatial dimensions, randomly select n indices
        flat_indices = torch.randperm(total_pixels)[:n]
        row_indices = flat_indices // r
        col_indices = flat_indices % r
        
        # Set selected pixels to white [0.5, 0.5, 0.5]
        s2[0, :, row_indices, col_indices] = 0.5
        
        # Compute distance
        dist = compute_pixel_distance(s1, s2)
        
        n_values.append(n)
        distances.append(dist)
        
        if n % 50 == 0 or n == 1 or n == total_pixels:
            print(f"  n={n:4d}: distance = {dist:.6f} (expected = {n * np.sqrt(3) / total_pixels:.6f})")
    
    # Convert to numpy for plotting
    n_values = np.array(n_values)
    distances = np.array(distances)
    
    # Expected values: (n / total_pixels) * sqrt(3)
    expected = n_values * np.sqrt(3) / total_pixels
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, distances, 'b-', linewidth=2, label='Computed distance')
    plt.plot(n_values, expected, 'r--', linewidth=2, label=f'Expected: (n/{total_pixels}) × √3')
    plt.xlabel('Number of white pixels (n)', fontsize=12)
    plt.ylabel('Pixel-wise distance', fontsize=12)
    plt.title(f'Distance between all-black image and image with n white pixels\n(r={r}, total pixels={total_pixels})', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save plot
    output_file = 'test_distance_plot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Display plot
    plt.show()
    
    # Verify the relationship
    print(f"\nVerification:")
    print(f"  Maximum distance (n={total_pixels}): {distances[-1]:.6f}")
    print(f"  Expected maximum: {np.sqrt(3):.6f}")
    print(f"  Ratio: {distances[-1] / np.sqrt(3):.6f}")


if __name__ == '__main__':
    main()

