# Start the real-data streams detached from any terminal (they keep running when the session or
# window that started them closes). Set RD_NO_COMMIT_GATE=1 to skip the free-commit gate. Each stream resumes from what is already on disk, so this is
# also the command to use after a reboot or a crash.
#
#   powershell -ExecutionPolicy Bypass -File code\launch_realdata.ps1            all streams and the digest loop
#   powershell -ExecutionPolicy Bypass -File code\launch_realdata.ps1 gpu gof    only some
#
# Progress: bash code/run_realdata.sh status, and logs\realdata\<stream>.log
# Stop: Stop-Process on the PIDs in logs\realdata\PIDS_<date>.txt and on any python.exe / Rscript.exe
# whose command line contains run_realdata or realdata_gof.

param([string[]]$Streams = @('gpu', 'cpu', 'tfmcpu', 'gof', 'gof_sepsis', 'digest'))
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$bash = 'C:\Program Files\Git\bin\bash.exe'
$rootPosix = '/' + $root.Substring(0, 1).ToLower() + ($root.Substring(2) -replace '\\', '/')
Remove-Item Env:REALDATA_SMOKE -ErrorAction SilentlyContinue
$out = @()
if ($env:RD_NO_COMMIT_GATE) { $gate = "RD_NO_COMMIT_GATE=$($env:RD_NO_COMMIT_GATE) " } else { $gate = "" }
foreach ($s in $Streams) {
    $script = if ($s -eq 'digest') { 'code/realdata_digest_loop.sh' } else { "code/run_realdata.sh $s" }
    $cmd = "cd $rootPosix && $($gate)bash $script >> logs/realdata/$s.out 2>&1"
    # created through WMI, so the stream is not in the job object of the shell (or terminal session)
    # that launched it and survives its end
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "`"$bash`" -c `"$cmd`""; CurrentDirectory = $root }
    $line = "{0} {1} PID {2} (WMI create, return {3})" -f (Get-Date -Format s), $s, $r.ProcessId, $r.ReturnValue
    $out += $line
    Write-Output $line
    Start-Sleep -Seconds 120          # stagger: each stream takes its memory before the next starts
}
$out | Add-Content (Join-Path $root ("logs\realdata\PIDS_{0}.txt" -f (Get-Date -Format yyyyMMdd)))
