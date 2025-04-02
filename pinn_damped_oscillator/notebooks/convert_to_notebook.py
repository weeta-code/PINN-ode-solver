import json
import re

def convert_md_to_notebook(md_file, notebook_file):
    """
    Convert a markdown file with code blocks to a Jupyter notebook format.
    """
    with open(md_file, 'r') as f:
        content = f.read()
    
    # Split the content by code blocks
    parts = re.split(r'```python\n(.*?)```', content, flags=re.DOTALL)
    
    cells = []
    
    # Process each part
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Markdown cell
            if part.strip():
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": part.split('\n')
                })
        else:  # Code cell
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": part.split('\n')
            })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write the notebook
    with open(notebook_file, 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    convert_md_to_notebook('/Users/victordesouza/Documents/PINN ode solver/pinn_damped_oscillator/notebooks/pinn_damped_oscillator.md', 'pinn_damped_oscillator.ipynb')
    print("Conversion complete!") 