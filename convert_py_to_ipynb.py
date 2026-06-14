import nbformat
import sys
from pathlib import Path

def convert_py_to_ipynb(py_path, ipynb_path):
    py_path = Path(py_path)
    ipynb_path = Path(ipynb_path)
    
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split by cells. We support both '# %%' and '# In[ ]:' format
    lines = content.split('\n')
    cells = []
    current_cell_type = 'code'
    current_cell_lines = []
    
    for line in lines:
        if line.startswith('# %%') or line.startswith('# In['):
            # Save previous cell
            if current_cell_lines:
                cell_content = '\n'.join(current_cell_lines).strip()
                if current_cell_type == 'markdown':
                    cells.append(nbformat.v4.new_markdown_cell(cell_content))
                else:
                    cells.append(nbformat.v4.new_code_cell(cell_content))
                current_cell_lines = []
            
            # Determine new cell type
            if 'markdown' in line.lower():
                current_cell_type = 'markdown'
            else:
                current_cell_type = 'code'
        else:
            # If the line is a commented line at the start of a markdown cell, strip the leading '# '
            if current_cell_type == 'markdown' and line.startswith('#'):
                # Strip leading '#' and one optional space
                line = line[1:]
                if line.startswith(' '):
                    line = line[1:]
            current_cell_lines.append(line)
            
    # Save the last cell
    if current_cell_lines:
        cell_content = '\n'.join(current_cell_lines).strip()
        if current_cell_type == 'markdown':
            cells.append(nbformat.v4.new_markdown_cell(cell_content))
        else:
            cells.append(nbformat.v4.new_code_cell(cell_content))
            
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully converted {py_path} to {ipynb_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python convert_py_to_ipynb.py <input.py> <output.ipynb>")
        sys.exit(1)
    convert_py_to_ipynb(sys.argv[1], sys.argv[2])
