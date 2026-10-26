param(
    [string]$InputDocx,
    [string]$OutputPdf,
    [int]$CloseEmptyPid = 0,
    [string]$ExpectedCreation = ''
)
$ErrorActionPreference = 'Stop'
if ($CloseEmptyPid -gt 0) {
    $application = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
    $wordProcesses = @(Get-Process -Name WINWORD)
    if ($wordProcesses.Count -ne 1) { throw 'Leave Word open: multiple instances prevent unique ownership' }
    $process = $wordProcesses[0]
    if ($process.Id -ne $CloseEmptyPid -or $application.Documents.Count -ne 0 -or
        [Math]::Abs(($process.StartTime.ToUniversalTime() - [DateTimeOffset]::Parse($ExpectedCreation).UtcDateTime).TotalMilliseconds) -gt 2) {
        throw 'Leave Word open: it is not the verified empty export instance'
    }
    $application.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($application)
    return
}
$application = New-Object -ComObject Word.Application
$application.Visible = $false
$application.DisplayAlerts = 0
$document = $null
try {
    $document = $application.Documents.Open((Resolve-Path -LiteralPath $InputDocx).Path, $false, $true)
    $document.Repaginate()
    $document.ExportAsFixedFormat([IO.Path]::GetFullPath($OutputPdf), 17)
} finally {
    if ($null -ne $document) { $document.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    $application.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($application)
}
