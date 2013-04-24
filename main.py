import prody as pr
import math
import numpy as np
import os
import sheets

DONOR_ACCEPTOR_MAXDISTANCE = 3.5
HYDROGEN_ACCEPTOR_MAXDISTANCE = 2.5
DHA_ANGLE = 90
DAB_ANGLE = 90
HAB_ANGLE = 90

#COLUMNS FOR DATA
#STRUCTURE INDICES ARE: 0=ALPHA, 1=310ALPHA, 2=PARALLEL, 3=ANTIPARALLEL
#COLUMN[STRUCTURE][1-N]
COLUMN_D_ON  = [[],[],[],[]] 
COLUMN_D_OH  = [[],[],[],[]] 
COLUMN_A_NHO = [[],[],[],[]] 
COLUMN_A_HOC = [[],[],[],[]] 
COLUMN_BETA  = [[],[],[],[]] 
COLUMN_GAMMA = [[],[],[],[]]
COLUMN_PHI   = [[],[],[],[]]
COLUMN_PSI   = [[],[],[],[]]

#COLUMNS FOR # OF H-BONDS, SAME STRUCTURE AS ABOVE
NUM_H_BONDS = [0,0,0,0]

def getHforAtom(appf, anAtom):
    if(anAtom.getResnum() < 1):
        return None
    resnumneighbors = appf.select('resnum '+str(anAtom.getResnum()))
    curMin = 100000
    target = None
    for at in resnumneighbors:
        if(at.getElement() == 'H'):
            #FILTER OTHER HYDROGENS
            dist = pr.calcDistance(at,anAtom)
            if(dist <= curMin):
            #print pr.calcDistance(at,anAtom), at.getResnum(), at.getName()
                curMin = dist
                target = at
    return target

def getAntecedent (appf, anAtom):
    if(anAtom.getResnum() < 1):
        return None
    aminoGroup = appf.select('resnum ' + str(anAtom.getResnum()))
    for at in aminoGroup:
        if(at.getName() == 'C'):
            return at
    return None

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
        #print out
        ho_ip = np.subtract(ho,np.multiply(n1_unit,out))
        ang = np.linalg.norm(ho_ip)/np.linalg.norm(ho)
        ang = math.acos(ang)
        #print "O"
        #print oCoords
        #print "C"
        #print cCoords
        #print "ho_ip"
        #print ho_ip
        #print np.add(ho_ip,oCoords)
        # we can think of H-O vector as the normal vector to a plane
        #ang = math.acos(np.dot(ho,n1)/(np.linalg.norm(ho)*np.linalg.norm(n1)))
        ang = ang*180/math.pi
        if(out < 0):
            ang = ang * -1
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
    n2 = np.cross(np.multiply(n1_unit,out),oc)
    #print n2
    ho_ip = np.subtract(ho,np.multiply(n1_unit,out))
    test = np.dot(n2,ho_ip)
    #print test
    ang = hproj/np.linalg.norm(ho_ip)
    ang = math.acos(ang)
    ang = ang*180/math.pi
    #if(test < 0):
    #    ang = ang * -1
    return ang
    
# find the number of models in a file
def getNumMdls(pfile):
    fp = open(pfile,'r')
    models = 0
    for line in fp:
        #print line[0:4]
        if(line[0:5] == "MODEL"):
            models = models+1
    fp.close()
    if(models == 0):
        return 1
    return models

#NOTE: WE WILL RUN IN BATCHES OF SIMILAR RESOLUTION
#pfile = '/Users/fsimon/Google Drive/School/Winter_13/Digital Biology/Project/DigBioProj_One/test.pdb'
#pfile = '1A6S_A_H.pdb'

def checkHBonds(appf,no_d, ox_a, ssindex):
    #1 Dist(don, acc) < 3.5
    da_dist = pr.calcDistance( no_d , ox_a )
    if( da_dist >= DONOR_ACCEPTOR_MAXDISTANCE ):
        return

    h_don = getHforAtom(appf,no_d)
    if(h_don == None):
        return
    #2 Dist(h_don, acc) < 3.5
    ha_dist  = pr.calcDistance(h_don, ox_a)
    if( ha_dist >=  HYDROGEN_ACCEPTOR_MAXDISTANCE):
        return
            
    #3 Angle(don, h_don, acc) > 90
    dha_ang = pr.calcAngle( no_d, h_don, ox_a )
    if( dha_ang < DHA_ANGLE ):
        return
            
    #4 Angle(don, acc, acc_ante) > 90
    #daa_ang =  pr.calcAngle( no_d, ox_a, acc_ante)
    #if( daa_ang < DAB_ANGLE ):
    #    return

    acc_ante = getAntecedent(appf, ox_a) #Get acc_ante
    if(acc_ante == None):
        return
    #5 Angle(h_don, acc, acc_ante) > 90
    haa_ang = pr.calcAngle( h_don, ox_a, acc_ante )
    if( haa_ang < HAB_ANGLE ):
        return

    #We have a valid H-Bond with no_d, ox_a, h_don
    #print 'HBond elements', h_don.getName(), no_d.getName(), ox_a.getName()
    #print 'DA dist', da_dist
    #print 'HA dist', ha_dist
    #print 'DHA ang', dha_ang
    #print 'DAA ang', daa_ang
    #print 'HAA ang', haa_ang
            
    beta_ang = getBetaAngle (appf,acc_ante,ox_a,h_don)
    gamm_ang = getGammaAngle(appf,acc_ante,ox_a,h_don)
        
    #PUT DATA INTO COLUMN DATA STRUCTURES
#    print ssindex
    #towrite = ",".join([str(h_don.getResnum()),str(h_don.getCoords())])
    #fp.write(towrite+"\r\n")
    NUM_H_BONDS[ssindex] += 1
    COLUMN_D_ON  [ssindex].append(da_dist)
    COLUMN_D_OH  [ssindex].append(ha_dist)
    COLUMN_A_NHO [ssindex].append(dha_ang)
    COLUMN_A_HOC [ssindex].append(haa_ang)
    COLUMN_BETA  [ssindex].append(beta_ang)
    COLUMN_GAMMA [ssindex].append(gamm_ang)
    return

def parseHelices(appf):
    # get secondary structure by aParsedPDBfile[i].getSecstr()
    nitrox_don = appf.select('element N O') #O can be donor in rare cases, the paper uses this convention
    oxygen_acc = appf.select('element O')
    for no_d in nitrox_don:
        n_secStruct = getSSIndex(no_d)
        # we're only concerned with helices
        if(n_secStruct < 0 or n_secStruct == 2):
            continue
        for ox_a in oxygen_acc:
            o_secStruct = getSSIndex(ox_a)
            if(o_secStruct != n_secStruct):
                continue
            ssindex = getSSIndex(ox_a) # get ssindex
            checkHBonds(appf,no_d,ox_a,ssindex)
    return

# do the sheets
def parseSheets(appf, los):
    strands = sheets.buildStrands(appf, los)
    numStrands = len(strands)
    for i in np.arange(numStrands-2):
        base = strands[i]
        comp = strands[i+1]
        if(comp.sense == 0):
            continue
        for no_d in base.getNAtoms():
            for ox_a in comp.getOAtoms():
                if(comp.getSense() > 0):
                    ssindex = 2
                else:
                    ssindex = 3
                checkHBonds(appf,no_d,ox_a,ssindex)
    return

# some mathy functions
def sumMeans(n1,n2,u1,u2):
    if(n2 == 0):
        return u1
    return (n1*u1+n2*u2) / (n1+n2)

def sumStds(n1,n2,u1,u2,s1,s2):
    if(n2 == 0):
        return s1
    x1 = (n1*s1*s1+n2*s2*s2) / (n1+n2)
    x2 = (n1*n2*((u1-u2)**2))/((n1+n2)**2)
    return math.sqrt(x1+x2)

# main function
def runThrough(pfile):
    # initial setup
    print "Running through " + pfile + "..."
    numMdls = getNumMdls(pfile)
    #print numMdls
    appf = pr.parsePDB(pfile, model=numMdls, secondary=True, chain='A', altLoc=False)
    los = sheets.initializeList(pfile)
    parseHelices(appf)
    if(los != None):
        parseSheets(appf,los)

    # get them means
    COLUMN_D_ON_AV  = [sum(x)/len(x) if len(x) > 0 else 0 for x in COLUMN_D_ON]
    COLUMN_D_OH_AV  = [sum(x)/len(x) if len(x) > 0 else 0 for x in COLUMN_D_OH]
    COLUMN_A_NHO_AV = [sum(x)/len(x) if len(x) > 0 else 0 for x in COLUMN_A_NHO]
    COLUMN_A_HOC_AV = [sum(x)/len(x) if len(x) > 0 else 0 for x in COLUMN_A_HOC]
    COLUMN_BETA_AV  = [sum(x)/len(x) if len(x) > 0 else 0 for x in COLUMN_BETA]
    COLUMN_GAMMA_AV = [sum(x)/len(x) if len(x) > 0 else 0 for x in COLUMN_GAMMA]

    # get the std devs, nasty shit
    COLUMN_D_ON_STD = [np.std(x) if len(x) > 0 else 0 for x in COLUMN_D_ON]
    COLUMN_D_OH_STD = [np.std(x) if len(x) > 0 else 0 for x in COLUMN_D_OH]
    COLUMN_A_NHO_STD = [np.std(x) if len(x) > 0 else 0 for x in COLUMN_A_NHO]
    COLUMN_A_HOC_STD = [np.std(x) if len(x) > 0 else 0 for x in COLUMN_A_HOC]
    COLUMN_BETA_STD = [np.std(x) if len(x) > 0 else 0 for x in COLUMN_BETA]
    COLUMN_GAMMA_STD = [np.std(x) if len(x) > 0 else 0 for x in COLUMN_GAMMA]

    TABLE = [COLUMN_D_ON_AV, COLUMN_D_OH_AV, COLUMN_A_NHO_AV, COLUMN_A_HOC_AV, COLUMN_BETA_AV, COLUMN_GAMMA_AV]
    STDS = [COLUMN_D_ON_STD, COLUMN_D_OH_STD, COLUMN_A_NHO_STD, COLUMN_A_HOC_STD, COLUMN_BETA_STD, COLUMN_GAMMA_STD]
    #print '        D_ON          D_OH      ANGLE(NHO)    ANGLE(HOC)        BETA         GAMMA   '
    #print np.array(TABLE).T
    #print np.array(STDS).T
    return (TABLE,STDS)

##runThrough("1A2Z_A_H.pdb")
# code to run through all the files and stuff
AGGR_D_ON  = [0,0,0,0] 
AGGR_D_OH  = [0,0,0,0] 
AGGR_A_NHO = [0,0,0,0] 
AGGR_A_HOC = [0,0,0,0] 
AGGR_BETA  = [0,0,0,0] 
AGGR_GAMMA = [0,0,0,0]

STD_D_ON = [0,0,0,0]
STD_D_OH = [0,0,0,0]
STD_A_NHO = [0,0,0,0]
STD_A_HOC = [0,0,0,0]
STD_BETA = [0,0,0,0]
STD_GAMMA = [0,0,0,0]

LEN_D_ON = [0,0,0,0]
LEN_D_OH = [0,0,0,0]
LEN_A_NHO = [0,0,0,0]
LEN_A_HOC = [0,0,0,0]
LEN_BETA = [0,0,0,0]
LEN_GAMMA = [0,0,0,0]

pFiles = os.listdir(os.getcwd())
# filter for only reduced pdb files
pFiles = [x for x in pFiles if x[-6:] == "_H.pdb"]
#print pFiles
for pfile in pFiles:
    # reset everything
    COLUMN_D_ON  = [[],[],[],[]] 
    COLUMN_D_OH  = [[],[],[],[]] 
    COLUMN_A_NHO = [[],[],[],[]] 
    COLUMN_A_HOC = [[],[],[],[]] 
    COLUMN_BETA  = [[],[],[],[]] 
    COLUMN_GAMMA = [[],[],[],[]]
    try:
        (results, stds) = runThrough(pfile)
    except (AttributeError,ValueError):
        continue
    # aggregate that shit
    AGGR_D_ON = [sumMeans(LEN_D_ON[i],len(COLUMN_D_ON[i]),AGGR_D_ON[i],
                          results[0][i]) for i in range(4)]
    AGGR_D_OH = [sumMeans(LEN_D_OH[i],len(COLUMN_D_OH[i]),AGGR_D_OH[i],
                          results[1][i])  for i in range(4)]
    AGGR_A_NHO = [sumMeans(LEN_A_NHO[i],len(COLUMN_A_NHO[i]),AGGR_A_NHO[i],
                          results[2][i])  for i in range(4)]
    AGGR_A_HOC = [sumMeans(LEN_A_HOC[i],len(COLUMN_A_HOC[i]),AGGR_A_HOC[i],
                          results[3][i])  for i in range(4)]
    AGGR_BETA = [sumMeans(LEN_BETA[i],len(COLUMN_BETA[i]),AGGR_BETA[i],
                          results[4][i])  for i in range(4)]
    AGGR_GAMMA = [sumMeans(LEN_GAMMA[i],len(COLUMN_GAMMA[i]),AGGR_GAMMA[i],
                          results[5][i])  for i in range(4)]

    STD_D_ON = [sumStds(LEN_D_ON[i],len(COLUMN_D_ON[i]),AGGR_D_ON[i],
                    results[0][i],STD_D_ON[i],stds[0][i]) for i in range(4)]
    STD_D_OH = [sumStds(LEN_D_OH[i],len(COLUMN_D_OH[i]),AGGR_D_OH[i],
                    results[1][i],STD_D_OH[i],stds[1][i]) for i in range(4)]
    STD_A_NHO = [sumStds(LEN_A_NHO[i],len(COLUMN_A_NHO[i]),AGGR_A_NHO[i],
                    results[2][i],STD_A_NHO[i],stds[2][i]) for i in range(4)]
    STD_A_HOC = [sumStds(LEN_A_HOC[i],len(COLUMN_A_HOC[i]),AGGR_A_HOC[i],
                    results[3][i],STD_A_HOC[i],stds[3][i]) for i in range(4)]
    STD_BETA = [sumStds(LEN_BETA[i],len(COLUMN_BETA[i]),AGGR_BETA[i],
                    results[4][i],STD_BETA[i],stds[4][i]) for i in range(4)]
    STD_GAMMA = [sumStds(LEN_GAMMA[i],len(COLUMN_GAMMA[i]),AGGR_GAMMA[i],
                    results[5][i],STD_GAMMA[i],stds[5][i]) for i in range(4)]

    LEN_D_ON = [(LEN_D_ON[i] + len(COLUMN_D_ON[i])) for i in range(4)]
    LEN_D_OH = [(LEN_D_OH[i] + len(COLUMN_D_OH[i])) for i in range(4)]
    LEN_A_NHO = [(LEN_A_NHO[i] + len(COLUMN_A_NHO[i])) for i in range(4)]
    LEN_A_HOC = [(LEN_A_HOC[i] + len(COLUMN_A_HOC[i])) for i in range(4)]
    LEN_BETA = [(LEN_BETA[i] + len(COLUMN_BETA[i])) for i in range(4)]
    LEN_GAMMA = [(LEN_GAMMA[i] + len(COLUMN_GAMMA[i])) for i in range(4)]

print ' MEANS '
print '        D_ON          D_OH      ANGLE(NHO)    ANGLE(HOC)        BETA         GAMMA   '
TABLE = [AGGR_D_ON,AGGR_D_OH,AGGR_A_NHO,AGGR_A_HOC,AGGR_BETA,AGGR_GAMMA]
print np.array(TABLE).T
print ' STANDARD DEVIATIONS '
print '        D_ON          D_OH      ANGLE(NHO)    ANGLE(HOC)        BETA         GAMMA   '
STDS = [STD_D_ON,STD_D_OH,STD_A_NHO,STD_A_HOC,STD_BETA,STD_GAMMA]
print np.array(STDS).T

print 'NUMBER OF H-BONDS'
print np.array(NUM_H_BONDS).T
