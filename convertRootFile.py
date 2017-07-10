#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

import sys
from pfcalibration.rootTools import rootToPython

"""
    Convertir un fichier root en fichier binaire importable par pickle
    Methode 1 :
    - rendre le programme executable
      chmod +x convertRootFile.py
    - lancer le programme suivi du nom des fichers à convertir
      ./convertRootFile.py myfile1.root myfile2.root
    Methode 2 :
    - rendre le programme executable
      chmod +x convertRootFile.py
    - lancer le programme, le nom du fichier à convertir vous sera demandé
      ./convertRootFile.py

"""


for arg in sys.argv:
    splited = arg.split('.')
    if splited[len(splited)-1] == "root":
        print("Conversion of",arg)
        rootToPython(arg)
        print("file converted")

if len(sys.argv) <= 1:
    arg = input("File to convert : ")
    print("Conversion of",arg)
    rootToPython(arg)
    print("file converted")
