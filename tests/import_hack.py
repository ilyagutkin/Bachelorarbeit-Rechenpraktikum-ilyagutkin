"""
This appends the '../src' directory to the system path, allowing modules from that directory to be imported.
This allows to use updated versions of the modules without having to reinstall them.
"""
import sys
sys.path.append('../src')
sys.path.append('src')    #if tests are run from the root directory
import methodsnm