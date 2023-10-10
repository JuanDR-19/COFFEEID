import colorspacious as cs

class grain:
    GREEN = cs.cspace_convert([55.9, -15.2, 20,7], start={"name": "CIELab"}, end={"name": "sRGB1"})  ##INMADURO
    MUSTARD = cs.cspace_convert([66.2, -4.0, 28,1], start={"name": "CIELab"}, end={"name": "sRGB1"}) ##SEMI-INMADURO
    PINT =  cs.cspace_convert([66, 9.5, 23.6], start={"name": "CIELab"}, end={"name": "sRGB1"}) ##POR SER MADURO
    M1 = cs.cspace_convert([42.2, 26.9, 3.3], start={"name": "CIELab"}, end={"name": "sRGB1"}) ##GRANO UTIL
    M2 = cs.cspace_convert([39.4, 16.8, 3.3], start={"name": "CIELab"}, end={"name": "sRGB1"}) ##GRANO UTIL
    SM1 = cs.cspace_convert([35.4, 13.7, -0.3], start={"name": "CIELab"}, end={"name": "sRGB1"}) ##SOBREMADURADO
    SM2 = cs.cspace_convert([28.8, 8.6, -0.9], start={"name": "CIELab"}, end={"name": "sRGB1"}) ##SOBREMADURADO
    DRY = cs.cspace_convert([15.9, 1.9, 0.7], start={"name": "CIELab"}, end={"name": "sRGB1"}) ##SECO
    