import numpy as np
from os.path import isfile, join
import xml.etree.ElementTree as ET

tree = ET.parse('DataReleaseDec2011/XMLMergedFile.xml')
root = tree.getroot()

for text in root.iter('TEXT'):
    print text.attrib