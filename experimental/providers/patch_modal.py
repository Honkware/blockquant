import pathlib

content = pathlib.Path('modal_app.py').read_text()
lines = content.split('\n')

# Find line 35 (0-indexed: 34)
for i, line in enumerate(lines):
    if i == 34:  # line 35
        print(f"Line {i+1}: {line[:120]}...")
        # Insert after line 35
        new_line = '''        # Also patch .get() fallback defaults — attn.py and sliding_attn.py use these
        "python3 -c \\"import pathlib; d=pathlib.Path(\'/opt/conda/lib/python3.11/site-packages/exllamav3\'); [p.write_text(p.read_text().replace(\'.get(\\"attn_mode\\", \\"flash_attn_nc\\")\', \'.get(\\"attn_mode\\", \\"sdpa_nc\\")\')) for p in d.rglob(\'*.py\') if \'.get(\\"attn_mode\\", \\"flash_attn_nc\\")\' in p.read_text()]\\",",'''
        lines.insert(i+1, new_line)
        break

pathlib.Path('modal_app.py').write_text('\n'.join(lines))
print('Inserted build-time patch after line 35')
