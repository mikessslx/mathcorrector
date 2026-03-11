Add-Type @"
using System;
using System.Runtime.InteropServices;
public class PowerState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$ES_DISPLAY_REQUIRED = [uint32]0x00000002
$flags = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED)
[void][PowerState]::SetThreadExecutionState([uint32]$flags)
Write-Host "Keep-awake enabled. Press Ctrl+C to stop and restore defaults."

try {
    while($true) { Start-Sleep -Seconds 1 }
} finally {
    [void][PowerState]::SetThreadExecutionState([uint32]2147483648)
    Write-Host "System power settings restored."
}
