from src.MBstation import *

def CMS():
    globalDTheight = 1.3
    SLgap     = 28.7 - globalDTheight*8 # originally, it was 29 - globalDTheight*8  ????
    nDTMB1 = 47
    nDTMB2 = 59
    nDTMB3 = 73
    nDTMB4 = 102
    wheel = -2
    sector = 11
    
    MB1   = MBstation(wheel = wheel, sector = sector, nDTs = nDTMB1, MBtype="MB1", gap = SLgap, SLShift = 0.5, additional_cells = 0)
    MB2   = MBstation(wheel = wheel, sector = sector, nDTs = nDTMB2, MBtype="MB2", gap = SLgap, SLShift = 1.0, additional_cells = 0)
    MB3   = MBstation(wheel = wheel, sector = sector, nDTs = nDTMB3, MBtype="MB3", gap = SLgap, SLShift = 0.0, additional_cells = 0)
    MB4   = MBstation(wheel = wheel, sector = sector, nDTs = nDTMB4, MBtype="MB4", gap = SLgap, SLShift = 2.0, additional_cells = 0)
    
    stations = {"MB1" : MB1, "MB2" : MB2, "MB3" : MB3, "MB4" : MB4}
    return stations
