from pathlib import Path
p = Path('run_log.txt')
data = p.read_bytes()
text = data.decode('utf-16')
Path('decoded_run_log.txt').write_text(text, encoding='utf-8')
print('wrote decoded_run_log.txt')
