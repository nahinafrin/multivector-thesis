from pathlib import Path
src = Path('run_log.txt')
dst = Path('run_log_decoded.txt')
data = src.read_bytes()
text = data.decode('utf-16')
dst.write_text(text, encoding='utf-8')
print(dst)
