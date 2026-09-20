@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python -c "from pathlib import Path; import time; p=Path('scheduler_canary_heartbeat.log'); p.write_text('start\n', encoding='utf-8'); [p.open('a', encoding='utf-8').write(f'heartbeat {i+1}/30\n') or time.sleep(10) for i in range(30)]; p.open('a', encoding='utf-8').write('complete\n')" > scheduler_canary_stdout.log 2> scheduler_canary_stderr.log