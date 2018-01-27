#!/bin/bash
VENV_FOLDER=catalyst-venv
if [ ! -d $VENV_FOLDER ]; then
    virtualenv catalyst-venv -p /usr/bin/python2
    source $VENV_FOLDER/bin/activate
    pip install enigma-catalyst matplotlib TA-lib
else
    source $VENV_FOLDER/bin/activate
fi
