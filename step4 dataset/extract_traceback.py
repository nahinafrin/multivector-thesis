from pathlib import Path
p = Path('run_log.txt')
data = p.read_bytes()
print('size', len(data))
print(data[:1000])
