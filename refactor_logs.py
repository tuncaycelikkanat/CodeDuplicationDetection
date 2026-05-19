import os
import re

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return

    original_content = content

    if 'from utils.logger import Log' not in content:
        import_block_end = content.rfind('\nimport ')
        if import_block_end == -1:
            import_block_end = content.rfind('\nfrom ')
        
        if import_block_end != -1:
            next_line = content.find('\n', import_block_end + 1)
            content = content[:next_line] + "\nfrom utils.logger import Log\n" + content[next_line:]
        else:
            content = "from utils.logger import Log\n" + content

    content = re.sub(r'print\(\s*f?"--->\s*(.*?)"\s*\)', r'Log.step(f"\1")', content)
    content = re.sub(r'print\(\s*f?"\s*→\s*(.*?)"\s*\)', r'Log.substep(f"\1")', content)
    content = re.sub(r'print\(\s*f?"\s*✅\s*(.*?)"\s*\)', r'Log.success(f"\1")', content)
    content = re.sub(r'print\(\s*f?"\s*⚠️\s*(.*?)"\s*\)', r'Log.warning(f"\1")', content)
    content = re.sub(r'print\(\s*f?"\s*🚀\s*(.*?)"\s*\)', r'Log.step(f"\1")', content)

    # Clean up any f'f"..."' syntax that might have occurred if there were no formatting vars initially but we added f""
    content = re.sub(r'Log\.step\(f"([^"{}]*)"\)', r'Log.step("\1")', content)
    content = re.sub(r'Log\.substep\(f"([^"{}]*)"\)', r'Log.substep("\1")', content)
    content = re.sub(r'Log\.success\(f"([^"{}]*)"\)', r'Log.success("\1")', content)
    content = re.sub(r'Log\.warning\(f"([^"{}]*)"\)', r'Log.warning("\1")', content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filepath}")

for root, dirs, files in os.walk('.'):
    if '.venv' in dirs:
        dirs.remove('.venv')
    if '.git' in dirs:
        dirs.remove('.git')
    for file in files:
        if file.endswith('.py') and file not in ['logger.py', 'refactor_logs.py']:
            process_file(os.path.join(root, file))
