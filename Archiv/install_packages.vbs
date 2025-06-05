Dim objShell, objFSO, scriptDirectory, tempBatPath, installerPath, downloadURL

Set objShell = WScript.CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Ermittelt den Pfad, in dem das Skript ausgeführt wird
scriptDirectory = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))

' Pfad zum Build-Tools-Installer
installerPath = scriptDirectory & "vs_buildtools.exe"

' URL zum Herunterladen der Visual Studio Build Tools
downloadURL = "https://aka.ms/vs/16/release/vs_buildtools.exe"

' Temporäres Batch-Skript
tempBatPath = scriptDirectory & "install_everything.bat"

' Überprüft, ob der Installer bereits heruntergeladen wurde
If Not objFSO.FileExists(installerPath) Then
    ' Herunterladen des Installers, wenn er nicht vorhanden ist
    With objFSO.CreateTextFile(tempBatPath, True)
        .WriteLine "@echo off"
        .WriteLine "echo Herunterladen der Visual Studio Build Tools..."
        .WriteLine "powershell -command ""(New-Object System.Net.WebClient).DownloadFile('" & downloadURL & "', '" & installerPath & "')"""
        .WriteLine "echo Installation der Visual Studio Build Tools..."
        .WriteLine """" & installerPath & """ --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --norestart"
        .WriteLine "echo Installation abgeschlossen."
        .WriteLine "pause"
    End With
Else
    ' Beginnt direkt mit der Installation, wenn der Installer schon vorhanden ist
    With objFSO.CreateTextFile(tempBatPath, True)
        .WriteLine "@echo off"
        .WriteLine "echo Installation der Visual Studio Build Tools..."
        .WriteLine """" & installerPath & """ --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --norestart"
        .WriteLine "echo Installation abgeschlossen."
        .WriteLine "pause"
    End With
End If

' Führt das Batch-Skript aus
objShell.Run "cmd /k """ & tempBatPath & """", 1, True

Set objShell = Nothing
Set objFSO = Nothing
