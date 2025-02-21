@echo off
pyinstaller --noconsole --onefile --name video-ppt-detector ^
    --add-data "src;src" ^
    --icon=app_icon.ico ^
    main.py 