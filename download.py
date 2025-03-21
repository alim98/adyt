import os
import shutil

source_dir = os.path.join('dyt', 'other_task')
if os.path.exists(source_dir):
    # Copy all contents from dyt/other_task to root directory
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        dest_item = os.path.join('.', item)
        
        if os.path.isdir(source_item):
            if os.path.exists(dest_item):
                shutil.rmtree(dest_item)
            shutil.copytree(source_item, dest_item)
        else:
            if os.path.exists(dest_item):
                os.remove(dest_item)
            shutil.copy2(source_item, dest_item)
