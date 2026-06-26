$ErrorActionPreference = 'Continue'
$py = 'C:\Users\zerg\AppData\Local\Programs\Python\Python312\python.exe'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:E17B_MAXDOCS = '0'
$env:E17B_ENCODERS = 'intfloat/multilingual-e5-small,intfloat/multilingual-e5-base'
$env:E17B_LANGS = 'sw,bn'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
Set-Location 'D:\PHD\research\RES1\shard'
$o = 'D:\PHD\research\RES1\results\exp17b_outputs'
$jp = "$o\exp17b_miracl.json"
$lg = "$o\launcher.log"
"[task] LOOP START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Append -Encoding utf8 $lg
for ($i = 1; $i -le 30; $i++) {
  "[task] attempt $i start $(Get-Date -Format 'HH:mm:ss')" | Out-File -Append -Encoding utf8 $lg
  & $py exp17b_miracl.py *>> "$o\py_stdout.log"
  $n = 0
  if (Test-Path $jp) {
    try { $n = [int](& $py -c "import json;print(len(json.load(open(r'$jp',encoding='utf-8'))['results']))") } catch { $n = 0 }
  }
  "[task] attempt $i ended $(Get-Date -Format 'HH:mm:ss'): $n/4 cells" | Out-File -Append -Encoding utf8 $lg
  if ($n -ge 4) { "[task] ALL 4 CELLS DONE" | Out-File -Append -Encoding utf8 $lg; break }
}
"[task] LOOP EXIT $(Get-Date -Format 'HH:mm:ss')" | Out-File -Append -Encoding utf8 $lg
