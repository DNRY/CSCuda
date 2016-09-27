$root = (split-path -parent $MyInvocation.MyCommand.Definition)
$version = [System.Reflection.Assembly]::LoadFile("$root\CSCuda\bin\Release\CSCuda.dll").GetName().Version
$versionStr = "{0}.{1}.{2}" -f ($version.Major, $version.Minor, $version.Build)

Write-Host "Setting .nuspec version tag to $versionStr"

$content = (Get-Content $root\CSCuda.nuspec) 
$content = $content -replace '\$version\$',$versionStr

$content | Out-File $root\CSCuda.compiled.nuspec

& nuget pack $root\CSCuda.compiled.nuspec