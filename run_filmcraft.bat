@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo FilmCraft を起動しています...
start "" "target\debug\filmcraft.exe" "..\..\movie\movie_creator3\samples\demo_project\demo_timeline.fcproj"
