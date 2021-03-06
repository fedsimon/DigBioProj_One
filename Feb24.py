import prody as pr
import math
import numpy as np

DONOR_ACCEPTOR_MAXDISTANCE = 3.5
HYDROGEN_ACCEPTOR_MAXDISTANCE = 2.5
DHA_ANGLE = 90
DAB_ANGLE = 90
HAB_ANGLE = 90

#COLUMNS FOR DATA
#STRUCTURE INDICES ARE: 0=ALPHA, 1=310ALPHA, 2=PARALLEL, 3=ANTIPARALLEL
#COLUMN[STRUCTURE][1-N]
COLUMN_D_ON  = [[],[],[]] # Convert to [[],[],[],[]] when we separate parallel and antiparallel
COLUMN_D_OH  = [[],[],[]] # Convert to [[],[],[],[]] when we separate parallel and antiparallel
COLUMN_A_NHO = [[],[],[]] # Convert to [[],[],[],[]] when we separate parallel and antiparallel
COLUMN_A_HOC = [[],[],[]] # Convert to [[],[],[],[]] when we separate parallel and antiparallel
COLUMN_BETA  = [[],[],[]] # Convert to [[],[],[],[]] when we separate parallel and antiparallel
COLUMN_GAMMA = [[],[],[]] # Convert to [[],[],[],[]] when we separate parallel and antiparallel


def getHforAtom(appf, anAtom):
        resnumneighbors = appf.select('resnum '+str(anAtom.getResnum()))
        for at in resnumneighbors:
            if(at.getElement() == 'H'):
            #FILTER OTHER HYDROGENS
                    return at

"""def getHforAtom(nAtom, oAtom, cAtom):
        # create hydrogen
        ag = oAtom.getAtomGroup()
        acsi = oATom.getACSIndex()
        H = pr.Atom(ag,acsi,'-1')
        # get coordinates
        nCoords = nAtom.getCoords()
        oCoords = oAtom.getCoords()
        cCoords = cAtom.getCoords()
        # C-O vector
        co = np.subtract(cCoords,oCoords)
        adjustment = np.divide(co,np.linalg.norm(co))
        hCoords = np.add(nCoords,adjustment)
        H.setCoords(hCoords)
        return H"""

def getAntecedent (appf, anAtom):
    aminoGroup = appf.select('resnum ' + str(anAtom.getResnum()))
    for at in aminoGroup:
        if(at.getName() == 'C'):
            return at

def getSSIndex(acc_ante):
    #RETURN 0=ALPHA, 1=310ALPHA, 2=PARALLEL, 3=ANTIPARALLEL
    #DSSP (Pauli) structure convention
    ante_str = acc_ante.getSecstr()
    if( ante_str == 'H' ):
        #alpha
        return 0
    if( ante_str == 'G' ):
        #310
        return 1
    if( ante_str == 'E' ):
        #BOTH PARALLEL AND ANTIPARALLEL
        return 2
    return -1

def getBetaAngle(appf,cAtom,oAtom,hAtom):
    # first determine the nAtom
    aminoGroup = appf.select('resnum ' + str(cAtom.getResnum()))
    for at in aminoGroup:
        if(at.getName() == 'N'):
            nAtom = at
        # get coordinates
        cCoords = cAtom.getCoords()
        oCoords = oAtom.getCoords()
        hCoords = hAtom.getCoords()
        nCoords = nAtom.getCoords()
    # get relevant vectors
    oc = np.subtract(oCoords,cCoords)
    nc = np.subtract(nCoords,cCoords)
    ho = np.subtract(hCoords,oCoords)
        # find norm vector to N-C-O plane
    n1 = np.cross(oc,nc)
    n1_unit = np.divide(n1,np.linalg.norm(n1))
    # find projection of H-O onto plane
    out = np.dot(ho,n1_unit)
    ho_ip = np.subtract(ho,np.multiply(n1_unit,out))
    ang = np.linalg.norm(ho_ip)/np.linalg.norm(ho)
    ang = math.acos(ang)
    # we can think of H-O vector as the normal vector to a plane
    #ang = math.acos(np.dot(ho,n1)/(np.linalg.norm(ho)*np.linalg.norm(n1)))
    ang = ang*180/math.pi
    # then we just need to adjust output by 90 degrees to get correct answer
    return ang

def getGammaAngle(appf,cAtom,oAtom,hAtom):
    # first determine the nAtom
    aminoGroup = appf.select('resnum ' + str(cAtom.getResnum()))
    for at in aminoGroup:
        if(at.getName() == 'N'):
            nAtom = at
        # get coordinates
        cCoords = cAtom.getCoords()
        oCoords = oAtom.getCoords()
        hCoords = hAtom.getCoords()
        nCoords = nAtom.getCoords()
    # get necessary vectors
    oc = np.subtract(oCoords,cCoords)
    nc = np.subtract(nCoords,cCoords)
    ho = np.subtract(hCoords,oCoords)
    n1 = np.cross(oc,nc)
    n1_unit = np.divide(n1,np.linalg.norm(n1))
    # get projection of H-O in O-C direction
    oc_unit = np.divide(oc,np.linalg.norm(oc))
    #print oc_unit
    hproj = np.dot(ho,oc_unit)
    # get projection of H-O onto N-C-O plane
    out = np.dot(ho,n1_unit)
    ho_ip = np.subtract(ho,np.multiply(n1_unit,out))
    ang = hproj/np.linalg.norm(ho_ip)
    ang = math.acos(ang)
    ang = ang*180/math.pi
    return ang*-1
    
# find the number of models in a file
def getNumMdl(pfile):
    fp = open(pfile,'r')
    models = 1
    for line in fp:
        if(line[0:5] == "NUMMDL"):
            models = int(line[10:13])
    return models

#NOTE: WE WILL RUN IN BATCHES OF SIMILAR RESOLUTION
#pfile = '/Users/fsimon/Google Drive/School/Winter_13/Digital Biology/Project/DigBioProj_One/test.pdb'
#pfile = '1A6S_A_H.pdb'

# main function
def runThrough(pfile):
    appf = pr.parsePDB(pfile, model=1, secondary=True, chain='A', altLoc=False)
    # get secondary structure by aParsedPDBfile[i].getSecstr()
    nitrox_don = appf.select('element N O') #O can be donor in rare cases, the paper uses this convention
    oxygen_acc = appf.select('element O')
    fp = open('old-H.txt','wb')
    for no_d in nitrox_don:
        n_secStruct = getSSIndex(no_d)
        if(n_secStruct < 0):
            continue
        for ox_a in oxygen_acc:
            o_secStruct = getSSIndex(ox_a)
            if(o_secStruct != n_secStruct):
                continue
        #1 Dist(don, acc) < 3.5
            da_dist = pr.calcDistance( no_d , ox_a )
            if( da_dist >= DONOR_ACCEPTOR_MAXDISTANCE ):
                continue

        #2 Dist(h_don, acc) < 3.5
            h_don = getHforAtom( appf, no_d ) #Get h_don
            ha_dist  = pr.calcDistance(h_don, ox_a)
            if( ha_dist >=  HYDROGEN_ACCEPTOR_MAXDISTANCE):
                continue
            
        #3 Angle(don, h_don, acc) > 90
            dha_ang = pr.calcAngle( no_d, h_don, ox_a )
            if( dha_ang < DHA_ANGLE ):
                continue
            
        #4 Angle(don, acc, acc_ante) > 90
            acc_ante = getAntecedent(appf, ox_a) #Get acc_ante
            daa_ang =  pr.calcAngle( no_d, ox_a, acc_ante)
            if( daa_ang < DAB_ANGLE ):
                continue
            
        #5 Angle(h_don, acc, acc_ante) > 90
            haa_ang = pr.calcAngle( h_don, ox_a, acc_ante )
            if( haa_ang < HAB_ANGLE ):
                continue

        #We have a valid H-Bond with no_d, ox_a, h_don
        #       print 'HBond elements', h_don.getName(), no_d.getName(), ox_a.getName()
        #       print 'DA dist', da_dist
        #       print 'HA dist', ha_dist
        #       print 'DHA ang', dha_ang
        #       print 'DAA ang', daa_ang
        #       print 'HAA ang', haa_ang
            
        #place holders for beta and gamma for now
        #   beta_ang = 0
        #   gamm_ang = 0
            beta_ang = getBetaAngle (appf,acc_ante,ox_a,h_don)
            gamm_ang = getGammaAngle(appf,acc_ante,ox_a,h_don)
        #PUT DATA INTO COLUMN DATA STRUCTURES
            ssindex = getSSIndex(acc_ante)
        #if (ssindex == -1):
            #Not a structure we need
            #   continue
            towrite = ",".join([str(h_don.getResnum()),str(h_don.getCoords())])
            fp.write(towrite+"\r\n")
            COLUMN_D_ON  [ssindex].append(da_dist)
            COLUMN_D_OH  [ssindex].append(ha_dist)
            COLUMN_A_NHO [ssindex].append(dha_ang)
            COLUMN_A_HOC [ssindex].append(haa_ang)
            COLUMN_BETA  [ssindex].append(beta_ang)
            COLUMN_GAMMA [ssindex].append(gamm_ang)
    
    fp.close()
    COLUMN_D_ON_AV  = [sum(x)/len(x) for x in COLUMN_D_ON if len(x) > 0]
    COLUMN_D_OH_AV  = [sum(x)/len(x) for x in COLUMN_D_OH if len(x) > 0]
    COLUMN_A_NHO_AV = [sum(x)/len(x) for x in COLUMN_A_NHO if len(x) > 0]
    COLUMN_A_HOC_AV = [sum(x)/len(x) for x in COLUMN_A_HOC if len(x) > 0]
    COLUMN_BETA_AV  = [sum(x)/len(x) for x in COLUMN_BETA if len(x) > 0]
    COLUMN_GAMMA_AV = [sum(x)/len(x) for x in COLUMN_GAMMA if len(x) > 0]

    TABLE = [COLUMN_D_ON_AV, COLUMN_D_OH_AV, COLUMN_A_NHO_AV, COLUMN_A_HOC_AV, COLUMN_BETA_AV, COLUMN_GAMMA_AV]
    print '        D_ON          D_OH      ANGLE(NHO)    ANGLE(HOC)        BETA         GAMMA   '
    print np.array(TABLE).T

runThrough('1A2Z_A_H.pdb')
