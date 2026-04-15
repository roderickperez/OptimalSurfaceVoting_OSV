#!/usr/bin/env python3
"""
Extract training history from the full-bandwidth PyTorch training notebook
and save it as NPZ for comparison with multi-spectral training.

Run this ONCE after training the full-bandwidth model.
"""

import os
import json
import numpy as np

# Path to the training notebook
NOTEBOOK_PATH = "/home/roderickperez/DataScienceProjects/faultSeg/faultSeg_2019/faultSeg_2019_pyTorch/train_newModel_pyTorch.ipynb"
OUTPUT_DIR = "/home/roderickperez/DataScienceProjects/faultSeg/faultSeg_2019/faultSeg_2019_pyTorch/output/history_plots"

def extract_history_from_notebook():
    """Extract training history from notebook execution outputs."""

    # Read notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Search for the training cell that has history data
    history = None

    for cell in nb['cells']:
        # Look for cells that might contain history dictionary
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))

            # Check if this is the training loop cell
            if 'history = {' in source and "'loss'" in source and "'accuracy'" in source:
                print("✓ Found training cell with history initialization")

                # Initialize empty history
                history = {
                    'loss': [],
                    'val_loss': [],
                    'accuracy': [],
                    'val_accuracy': []
                }

                # Look at outputs to find printed epoch results
                outputs = cell.get('outputs', [])
                for output in outputs:
                    if output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                        text = output.get('text', '')
                        if isinstance(text, list):
                            text = ''.join(text)

                        # Parse epoch lines like:
                        # "Epoch 01/25 | loss 0.0778 – acc 0.6456 | val_loss 0.0622 – val_acc 0.8022"
                        for line in text.split('\n'):
                            if 'Epoch' in line and 'loss' in line and 'val_loss' in line:
                                try:
                                    parts = line.split('|')
                                    if len(parts) >= 3:
                                        # Parse train metrics
                                        train_part = parts[1].strip()
                                        train_loss = float(train_part.split('loss')[1].split('–')[0].strip())
                                        train_acc = float(train_part.split('acc')[1].strip())

                                        # Parse val metrics
                                        val_part = parts[2].strip()
                                        val_loss = float(val_part.split('val_loss')[1].split('–')[0].strip())
                                        val_acc = float(val_part.split('val_acc')[1].strip())

                                        history['loss'].append(train_loss)
                                        history['accuracy'].append(train_acc)
                                        history['val_loss'].append(val_loss)
                                        history['val_accuracy'].append(val_acc)
                                except Exception as e:
                                    continue

                if len(history['loss']) > 0:
                    print(f"✓ Extracted {len(history['loss'])} epochs of training history")
                    return history

    return None

def main():
    print("=" * 60)
    print("  Extracting Full-Bandwidth Training History")
    print("=" * 60)

    history = extract_history_from_notebook()

    if history is None or len(history['loss']) == 0:
        print("\n❌ Could not extract history from notebook.")
        print("   Make sure the training notebook has been executed.")
        print("\n💡 Alternative: Manually create history NPZ from your records.")
        return

    # Save as NPZ
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "history_fullbw.npz")

    np.savez(
        output_path,
        loss=np.array(history['loss'], dtype=np.float32),
        val_loss=np.array(history['val_loss'], dtype=np.float32),
        accuracy=np.array(history['accuracy'], dtype=np.float32),
        val_accuracy=np.array(history['val_accuracy'], dtype=np.float32)
    )

    print(f"\n✅ Saved training history to: {output_path}")
    print(f"\n📊 Summary:")
    print(f"   - Epochs: {len(history['loss'])}")
    print(f"   - Final train loss: {history['loss'][-1]:.4f}")
    print(f"   - Final train acc:  {history['accuracy'][-1]:.4f}")
    print(f"   - Final val loss:   {history['val_loss'][-1]:.4f}")
    print(f"   - Final val acc:    {history['val_accuracy'][-1]:.4f}")
    print("\n🎯 Now you can run the multi-spectral training notebook!")
    print("   It will automatically include this baseline in the plots.")

if __name__ == "__main__":
    main()
