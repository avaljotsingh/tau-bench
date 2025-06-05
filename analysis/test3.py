from vulture import Vulture

v = Vulture()
with open('analysis/test1.py') as f:
    v.scan(f.read())
v.report()