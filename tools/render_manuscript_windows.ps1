param(
  [Parameter(Mandatory=$true)][string]$InputDocx,
  [Parameter(Mandatory=$true)][string]$OutputPdf
)
# Export only the requested document through a separate Word automation instance.
# Microsoft Word must be installed. The source is opened read-only.
$taskWord = New-Object -ComObject Word.Application
$taskWord.Visible = $false
$taskWord.DisplayAlerts = 0
try {
  $taskDoc = $taskWord.Documents.Open((Resolve-Path -LiteralPath $InputDocx).Path, $false, $true)
  $taskDoc.Repaginate()
  $taskDoc.ExportAsFixedFormat([System.IO.Path]::GetFullPath($OutputPdf), 17)
  $taskDoc.Close(0)
} finally {
  $taskWord.Quit(0)
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskWord) | Out-Null
}
