#!/bin/bash
pyinstaller --noconsole --onefile --name video-ppt-detector \
    --add-data "src:src" \
    --icon=app_icon.icns \
    main.py 