import sys

sys.path.append(".")
from asdc.sdc import reglo

r = reglo.Reglo(address="COM16")
r.stop()
