# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:39:39 2024

@author: Naz
"""

import os
import re
import sys
import sqlite3
from sqlite3 import Error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import shutil




def getFragments(filepath, bound, times=1, contains=""):
    boundaries = {        
        "bonds" : ("Optimized Parameters","GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad"),
        "energies" : ("Zero-point correction=","Axes restored to original set"),
        "frequencies" : ("Diagonal vibrational polarizability","Thermochemistry"),
        "nbo_summary" : ("Natural Bond Orbitals (Summary)", "Calling FoFJK,"),
        "nbo_fock" : ("E(2)  E(j)-E(i) F(i,j)","Natural Bond Orbitals (Summary)"),
        "nho_directionality" : ("NHO Directionality and \"Bond Bending\"","Second Order Perturbation Theory Analysis of Fock Matrix in NBO Basis"),
        "nbo_start" : ("NATURAL BOND ORBITAL ANALYSIS","NHO Directionality and \"Bond Bending\""),
        "npa_summary" : ("Natural Population","NATURAL BOND ORBITAL ANALYSIS"),
        "nao_occupancies" : ("NATURAL POPULATIONS:  Natural atomic orbital occupancies","Summary of Natural Population Analysis"),
        "mull_charges" : ("Mulliken charges","Sum of Mulliken charges"),
        "ao_scf" : ("Condensed to atoms (all electrons):","Mulliken charges"),
        "pa_scf" : ("analysis using the SCF","Condensed to atoms (all electrons)"),
        "coord" : ("Freq\\","@"),
        "termination" : ("",""),
        "hirshfeld" : ("Hirshfeld populations at iteration", "Approx polarizability"),
        "input" : ("Input orientation","GradGrad"),
        "standard" : ("Standard orientation","GradGrad"),
        "standard_short" : ("Standard orientation","Rotational constants"),
        "optimized" : ("Optimized Parameters","GradGradGradGradGradGradGradGradGrad")
    }
    
    result = []
    restart = False
    with open(filepath,"r") as file:            
        start = str(boundaries.get(bound)[0])
        finish = str(boundaries.get(bound)[1])
        buffer = []
        status = "search"
        addStatus = False
        counter = 0
        for line in file:
            if start in line:
                status = "write"
                counter = 0
            if contains in line:
                addStatus= True
            if finish in line:
                counter = counter+1
            if status == "write":
                if counter<=times:                      
                      buffer.append(line)   
                if (counter==times):
                    if addStatus == True:
                        result.append(buffer)                          
                    # print(len(buffer))
                    buffer = []
                    addStatus = False
                    status = "search" 
            if "FormBX had a problem" in line:
                restart = True                
    # print(len(result))
    return result, line



####   Find Distances   ###################

def getDistFromFrag(frag, atom1, atom2):
    dist = 0
    if atom1 < atom2:
        pat = "R({},{})".format(str(atom1),str(atom2))
    else:
        pat = "R({},{})".format(str(atom2),str(atom1))
    for i in range(0,len(frag)):
        if pat in frag[i]:
            result = re.search(r".+R[0-9]+\s+R\([0-9,]+\)\s+([0-9.]+).*",frag[i])
            dist = result.group(1)
    return dist



def getDistFromFile(filepath, atom1, atom2, times=1, contains=""):
    dist = []
    frag_list, term = getFragments(filepath, "standard", times, contains)   
    if (len(frag_list) > 0) and ("Normal termination of Gaussian 16" in term):
        for i in range(0,len(frag_list)):
            frag = frag_list[i]
            if len(frag) > 0:
                dist.append(getDistFromFrag(frag, atom1, atom2))
    else:
        dist = [0]     
    return dist


def getXYZMatrixFromFrag(frag):
    xyz = []
    start = False
    pat = r"\s+[0-9]+\s+([0-9]+)\s+[0-9]\s+(-?[0-9]+\.[0-9]+)\s+([0-9.-]+)\s+([0-9.-]+).*"
    for i in range(0,len(frag)):
        if "Standard orientation" in frag[i]:
            start = True                                      
        if start == True  and re.match(pat, frag[i]):
            result = re.search(pat,frag[i])
            xyz.append([float(result.group(2)),float(result.group(3)),float(result.group(4))])    
    return xyz




####   Compute Distances   ############################


def computeDistSum(coord, atom_list=[1,0],limit = 1.7):
    matrix = [] 
    dist_sum = 0
    if len(coord)>0:
        for i in range(0,len(coord)):
            if i in atom_list:
                matrix.append(coord[i])     
        matrix1 = np.array(matrix)      
        dist12 = distance.cdist(matrix1,matrix1)
        for i in range(0,len(dist12)):
            for j in range(0,i):
                if abs(dist12[i][j]) < limit:
                    dist_sum += abs(dist12[i][j])    
    return round(dist_sum,5)



def getDistSumFromFile(filepath, atom_list=[1,0],limit = 2.5,  times=1, contains=""):
    dist_sum = []
    atom_list = [x-1 for x in atom_list]
    frag_list, term = getFragments(filepath, "standard", times, contains)  
    # if (len(frag_list) > 0) and ("Normal termination of Gaussian 16" in term):        
    if (len(frag_list) > 0):        
        for i in range(0,len(frag_list)):
            xyz = getXYZMatrixFromFrag(frag_list[i])
            dist_sum.append(computeDistSum(xyz,atom_list,limit))           
    else:
       dist_sum = [0]
    return dist_sum




####   FMO   ##########################

def getLUMOFromFrag(frag):
    lumo = []
    for line in frag:
        if "Alpha virt. eigenvalues" in line:
            an_lumo = re.search(r"\sAlpha virt\. eigenvalues\s\-\-\s+([0-9-.]+)\s?\s?([0-9-.]+)?\s?\s?([0-9-.]+)?\s?\s?([0-9-.]+)?\s?\s?([0-9-.]+)?",line)
            for i in range(1,6):
                if an_lumo.group(i) is not None:
                    lumo.append(float(an_lumo.group(i)))
    return lumo

    

def getHOMOFromFrag(frag):
    homo = []
    for line in frag:
        if "Alpha  occ. eigenvalues" in line:
            an_homo = re.search(r"\sAlpha  occ\. eigenvalues\s\-\-\s+([0-9-.]+)\s{0,2}([0-9-.]+)?\s{0,2}([0-9-.]+)?\s{0,2}([0-9-.]+)?\s{0,2}([0-9-.]+)?",line)
            for i in range(1,6):
                if an_homo.group(i) is not None:
                    homo.append(float(an_homo.group(i)))

    return homo



def getFMOFromFile(filepath, times=1, contains="",  fmo_type="lumo", fmo_number=1):
    fmo = []
    # frag_list, term = getFragments(filepath, "standard", times, contains)       
    frag_list, term = getFragments(filepath, "pa_scf", times, contains)       
    # if (len(frag_list) > 0) and ("Normal termination of Gaussian 16" in term):
    if (len(frag_list) > 0) :
        for i in range(0,len(frag_list)):
            frag = frag_list[i]
            # print(frag)
            if len(frag) > 0 and fmo_type == "lumo":
               fmo.append(getLUMOFromFrag(frag)[fmo_number-1])
            if len(frag) > 0 and fmo_type == "homo":
                fmo.append(getHOMOFromFrag(frag)[-1*fmo_number])

    else:
        fmo = [0]
    return fmo[-1]


####   FMO from NBO   ##################


def getNBOFMOFromFrag(frag):
    
    homo_x = []
    homo_y = []
    lumo_x = []
    lumo_y = []
    pat1 = r"\s+[0-9]+\.\s+([BDCRLPY\*]+).?\([0-9 ]+\)\s[A-Za-z]+\s+([0-9]+)(\s-\s[A-Za-z]+\s+([0-9]+))?\s+([0-9.]+)\s+([0-9.-]+).*"
    if (len(frag) > 0):
        start = False
        for i in range(0,len(frag)):
            if "Natural Bond Orbitals (Summary)" in  frag[i]:
                start = True
            if start == True and re.match(pat1,frag[i]):
                result = re.search(pat1, frag[i])
                key = "{}_{}_{}".format(result.group(1),result.group(2),result.group(4))
                if result.group(1) in ["BD","LP"]:
                    homo_x.append(key)
                    homo_y.append(float(result.group(6)))
                if result.group(1) in ["BD*","LP*"]:
                    lumo_x.append(key)
                    lumo_y.append(float(result.group(6)))
        homo_table = pd.DataFrame(homo_y,homo_x).sort_values(by=[0],ascending=False)
        lumo_table = pd.DataFrame(lumo_y,lumo_x).sort_values(by=[0],ascending=True)
    else:
        homo_table = pd.DataFrame(["homo"],["0"])
        lumo_table = pd.DataFrame(["lumo"],["0"])
    return homo_table, lumo_table


def getNPOFMOFromFrag(frag):    
    homo = {}
    lumo = {}
    pat1 = r"\s+[0-9]+\s+[A-Z][a-z]?\s+([0-9]+)\s+([Spdfxyz\d]+)\s+(Val|Ryd)\(\s([0-9Spdf]+)\)\s+([0-9.]+)\s+([0-9.-]+)"
    if (len(frag) > 0):
        start = False
        for i in range(0,len(frag)):
            if "Natural atomic orbital occupancies" in  frag[i]:
                start = True
            if start == True and re.match(pat1,frag[i]):
                result = re.search(pat1, frag[i])
                key = "{}_{}_{}({})".format(result.group(1),result.group(2),result.group(3),result.group(4))
                if result.group(3) == "Val" and result.group(4) == "2p":
                    homo[key] = float(result.group(6))
                    # homo_y.append(float(result.group(6)))
                if result.group(3) == "Ryd":
                    lumo[key] = float(result.group(6))
                    # lumo_x.append(key)
                    # lumo_y.append(float(result.group(6)))
    #     homo_table = pd.DataFrame(homo_x,homo_y).sort_values(by=[0],ascending=False)
    #     lumo_table = pd.DataFrame(lumo_x,lumo_y).sort_values(by=[0],ascending=True)
    # else:
        homo_table = pd.DataFrame(["homo"],["0"])
        lumo_table = pd.DataFrame(["lumo"],["0"])
    return homo, lumo



def getNBOFMOFromFile(filepath, fmo="lumo", times=1, contains=""):
    nbo_fmo = []
    frag_list, term = frag_list, term = getFragments(filepath, "standard", times, contains)
    if (len(frag_list) > 0) and ("Normal termination of Gaussian 16" in term):
        for i in range(0,len(frag_list)):
            if len(frag_list[i]) > 0:
                homo,lumo = getNPOFMOFromFrag(frag_list[i])
                if fmo == "lumo":
                    nbo_fmo.append(lumo)
                if fmo == "homo":
                    nbo_fmo.append(homo)
                else:
                    nbo_fmo.append(0)
    else:
        nbo_fmo = [0]
    return nbo_fmo              


def getNPOFMOTableFromFile(filepath, fmo="lumo", times=1, contains=""):
    nbo_fmo = []
    nbo_title = []
    frag_list, term = frag_list, term = getFragments(filepath, "standard", times, contains)
    if (len(frag_list) > 0) and ("Normal termination of Gaussian 16" in term):
        for i in range(0,len(frag_list)):
            if len(frag_list[i]) > 0:
                homo,lumo = getNPOFMOFromFrag(frag_list[i])
                if fmo == "lumo":
                    raw = pd.DataFrame(lumo, index=[i])
                if fmo == "homo":
                    raw = pd.DataFrame(homo, index=[i])
                # else:
                #     nbo_fmo.append(0)
                if i==0:
                    fmo_table = raw
                if i>0:
                    fmo_table = pd.concat([fmo_table,raw], ignore_index = False)            
    return fmo_table



####  Hirsh  ######################


def getHirshFromFrag(frag, atom, withH = False):
    hirsh1 = 0
    hirsh2 = 0 
    pat1 = r"\s+([0-9]+)\s+[A-Z][a-z]?\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\s+.([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+).*"             
    pat2 = r"\s+([0-9]+)\s+[A-Z][a-z]?\s+([0-9.-]+)\s+([0-9.-]+).*"             
    for i in range(0,len(frag)):        
        if re.match(pat1,frag[i]): 
            result1 = re.search(pat1,frag[i])
            if int(result1.group(1)) in atom and float(result1.group(3))==0:                              
                hirsh1 += float(result1.group(2))       
        if re.match(pat2,frag[i]): 
            result2 = re.search(pat2,frag[i])
            if int(result2.group(1)) in atom:                              
                hirsh2 += float(result2.group(2))  
    if withH==True:
       hirsh = hirsh1
    else:
        hirsh = hirsh2
    return hirsh



def getHirshFromFile(filepath, atom, withH = False, times=1, contains=""):
    hirsh = []
    frag_list, term = getFragments(filepath, "standard", times, contains)
    print(len(frag_list))
    # if (len(frag_list) > 0) and ("Normal termination of Gaussian 16" in term):
    if (len(frag_list) > 0) :
        for i in range(0,len(frag_list)):
            frag = frag_list[i]
            if len(frag) > 0:
               hirsh.append(getHirshFromFrag(frag,atom, withH))
    else:
        hirsh = [0]
    return hirsh



####   SCF Energy   ######################

def getEnergyFromFrag(frag):
    energy = 0
    pat = r"\s+SCF\sDone:\s+E\(RM06\)\s=\s+([0-9.-]+)\s.*"
    for i in range(0,len(frag)):
        if re.match(pat,frag[i]):
          result = re.search(pat,frag[i])                            
          energy= float(result.group(1))
    return energy


def getEnergyFromFile(filepath, times=1, contains=""):
    energy = []
    frag_list, term = getFragments(filepath, "standard", times, contains)
    # if (len(frag_list) > 0) and ("Normal termination of Gaussian 16" in term):
    if (len(frag_list) > 0):
        for i in range(0,len(frag_list)):
            frag = frag_list[i]
            if len(frag) > 0:
               energy.append(getEnergyFromFrag(frag))
    else:
        energy = [0]
    return energy




####  FreeEnergy  ######################


def getFreeEnergyFromFrag(frag):
    freeEnergy = 0
    pat = r"\s+Sum of electronic and thermal Free Energies=\s+([0-9.-]+).*"
    for i in range(0,len(frag)):
        if re.match(pat,frag[i]):
            result = re.search(pat,frag[i])
            freeEnergy = float(result.group(1))
    return freeEnergy


def getFreeEnergyFromFile(filepath):
    freeEnergy = []
    frag_list, term = getFragments(filepath, "energies")
    # if len(frag_list)>0 and "Normal termination of Gaussian" in term:
    if len(frag_list)>0:
        for i in range(0,len(frag_list)):       
            freeEnergy.append(getFreeEnergyFromFrag(frag_list[i]))
    else:
        freeEnergy = [0]
    return freeEnergy


####   Files From Scan   ##################


def getICFromFrag(frag):
    IC = ""
    patR = r".*(R\([0-9]+,[0-9]+\))\s+([0-9.-]+).*"
    patA = r".*(A\([0-9]+,[0-9]+,[0-9]+\))\s+([0-9.-]+).*"
    patD = r".*(D\([0-9]+,[0-9]+,[0-9]+,[0-9]+\))\s+([0-9.-]+).*"
    pat = r".*([RAD]\(([0-9]+,?){2,4}\))\s+([0-9.-]+).*"
    for i in range(0,len(frag)):
        if re.match(patR, frag[i]):
            result = re.search(patR, frag[i]) 
            IC += "{}={}\n".format(result.group(1),result.group(2))
        if re.match(patA, frag[i]):
            result = re.search(patA, frag[i])
            IC += "{}={}\n".format(result.group(1),result.group(2))
        if re.match(patD, frag[i]):
            result = re.search(patD, frag[i])
            IC += "{}={}\n".format(result.group(1),result.group(2))

    return IC
    

def getXYZFromFrag(frag, atoms = [1]):
    xyz = ""
    atom = ""
    start = False
    pat = r"\s+([0-9]+)\s+([0-9]+)\s+[0-9]\s+(-?[0-9]+\.[0-9]+)\s+([0-9.-]+)\s+([0-9.-]+).*"
    for i in range(0,len(frag)):
        if "Standard orientation" in frag[i]:
            start = True                                      
        if start == True  and re.match(pat, frag[i]):
            result = re.search(pat,frag[i])
            # if result.group(1) in atoms:
            atomN = int(result.group(2))
            match (atomN):
                case 1 : atom = "H"
                case 6 : atom = "C"
                case 7 : atom = "N"
                case 8 : atom = "O"
                case 9 : atom = "F"
            xyz += "  {}      {}      {}      {}\n".format(atom,result.group(3),result.group(4),result.group(5))    
    return xyz
    



def getICListFromFile(filepath):
    molList = []
    frag_list, term = getFragments(filepath, "optimized")
    if (len(frag_list) > 0) and "Normal termination of Gaussian 16" in term:
        for i in range (0,len(frag_list)):
            molList.append(getICFromFrag(frag_list[i]))
    else:
        molList = [0]
    return molList



def getXYZListFromFile(filepath, atoms = [1]):
    molList = []
    frag_list, term = getFragments(filepath, "standard_short", times=2, contains="Optimized Parameters")
    if (len(frag_list) > 0) and "Normal termination of Gaussian 16" in term:
        for i in range (0,len(frag_list)):
            molList.append(getXYZFromFrag(frag_list[i], atoms))
    else:
        molList = [0]
    return molList



def createGaussFiles(in_file, folder, atoms = [1], templ_file="tmp.gjf",  name="file"):    
    IC_list = getXYZListFromFile(in_file, atoms)
    buffer = ""
    link = "\n--Link1--\n%nprocshared=16\n%mem=16GB\n%chk=name.chk\n# freq=noraman M06/6-31+G(d,p) geom=Check guess=read pop=(nbo,Hirshfeld)\n\n0 2\n  \n"
    if templ_file=="tmp.gjf":
        templ_file = os.path.join(folder,"tmp.gjf")
    with open(templ_file,"r") as file:
        buffer = file.readlines()
        buffer = "".join(buffer)
    if len(IC_list) > 0:
        for i in range(0,len(IC_list)):
          title = name+"_"+str(i)
          fn = os.path.join(folder,title+".gjf")
          buffer = buffer.replace("name",title)
          link = link.replace("name",title)
          with open(fn,"w") as new_file:
              new_file.writelines(buffer)
              new_file.writelines(IC_list[i])
              new_file.write(link)
              print(fn)
    else:
        print("The IC list is empty")   
        
        
#####   copy files from to   ###################


def copyFiles(folder_in, folder_out):
    for fol,fi,dir in os.walk(folder_in):
        for a in dir:
            if re.match(r"nu([0-9]+)\.log",a):                
                file =  os.path.basename(os.path.join(fol,a))
                print(os.path.join(fol,a))
                shutil.copy(os.path.join(fol,a),os.path.join(folder_out,file))

        
        
        
#####   append Data Base    ####################


def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Program connects to the db")
    except Error as e:
        print(f"The error '{e}' occured")
    return connection



def queryExe(connection, query):
    curs = connection.cursor()
    try:
        curs.execute(query)
        connection.commit()
    except Error as e:
        print(f"The error '{e}' occured")
        

def tableQuery(table_name, field):
    com_str = "CREATE TABLE IF NOT EXISTS " + str(table_name) + " (\n"
    # com_str += "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
    com_str += "  title TEXT NOT NULL UNIQUE,\n"
    # for title in all_fields:
    com_str += "  {} REAL,\n".format(field)
    
    com_str = com_str.rstrip(com_str[-1])
    com_str = com_str.rstrip(com_str[-1])
    com_str += "\n);"
    com_str = """{}\n""".format(com_str)
    print(com_str)
    return com_str
        


def joinTabes(db,new_table, tables_list, fields_list, common_field):
    connection = create_connection(db)
    query = "CREATE TABLE " + new_table + " AS\nSELECT "
    # query += "e_LUMO.title, LUMO,h_LUMO,HOMO,h_HOMO,Energy,h_Energy"
    for i in range (0,len(fields_list)):        
      query += fields_list[i] + ", "
    # query += "e_LUMO.title, e_LUMO.LUMO, e_h_LUMO.h_LUMO, e_HOMO.HOMO, e_h_HOMO.h_HOMO, e_Energy.Energy, e_h_Energy.h_Energy"
    query = query[:-2] + "\nFROM " + tables_list[0] + "\n"
    for i in range (1,len(tables_list)):
        query += "INNER JOIN " + tables_list[i]  + " ON " + tables_list[i] + """.""" + common_field + " = " + tables_list[0] + """.""" + common_field + "\n"
    query += ";"
    print(query)
    queryExe(connection, query)

  
    
  
def appendDataToDB(db_path, table_name, field_name, data_table):
    connection = create_connection(db_path)
    table_query = tableQuery(table_name,field_name)
    queryExe(connection, table_query)
    com_str = """INSERT INTO """+"{}".format(table_name)+""" (title,""" + str(field_name)+ """) VALUES \n"""
    for i in range(0,len(data_table)):
        # print(data_table.iloc[i,0])
        # print(data_table.iloc[i,1])
        # com_str += """(\""""+str(data_table.iloc[i,0])+"\", " + str(data_table.iloc[i,1])+"),\n"""
        if str(data_table.iloc[i,1]) != "nan":
            com_str += """(\""""+str(data_table.iloc[i,0])+"\", " + str(data_table.iloc[i,1])+"),\n"""
    final_str = com_str[:-2] + ";\n"
    print(final_str)
    queryExe(connection, final_str)
        
     
def tableFromDB(db):
    return pd.read_sql_query("SELECT * FROM Nucleophiles", create_connection(db))
    
        
        
#####   Files modification in folder    ###############       
        
def getXYZFromFile(filepath):
    molList = []
    frag_list, term = getFragments(filepath, "standard")
    # print(frag_list[-1])
    if (len(frag_list) > 0) and "Normal termination of Gaussian 16" in term:
        for i in range (0,len(frag_list)):
            molList.append(getXYZFromFrag(frag_list[i]))
    else:
        molList = [0]
    return molList[-1]        
     
   
          
def modifiedFilesInFolder(folder_in, folder_out, templ_file="tmp.gjf",  pat="", add_to_name="tol"):   
    file_list = os.scandir(folder_in)
    for file in file_list:
        if file.is_file() and re.match(pat, file.name):
            file_path = os.path.join(folder_in,file.name)
            result = re.search(pat,file.name)
            print(file_path)
            IC_list = getXYZFromFile(file)  
            buffer = ""  
            if templ_file=="tmp.gjf":
                templ_file = os.path.join(folder_out,"tmp.gjf")
            with open(templ_file,"r") as file_tmp:
                buffer = file_tmp.readlines()
                buffer = "".join(buffer)
            title = result.group(1)+"_"+add_to_name
            fn = os.path.join(folder_out,title+".gjf")
            buffer = buffer.replace("name",title)
            with open(fn,"w") as new_file:
              new_file.writelines(buffer)
              new_file.writelines(IC_list)
              new_file.writelines(" ")

    else:
        print("The IC list is empty")  





####  VISUALIZE RESULTS   #################


def createTable(x,y):
    
    return pd.DataFrame({"name": x, "value" : y})
    

def dataFromFolder(folder, data_type,  pat="", comp_dict = {}, a=-1, b=-1):
    x = []
    y = []
    file_list = os.scandir(folder)
    for file in file_list:
        if file.is_file() and re.match(pat, file.name):
            result= re.search(pat, file.name)
            x.append(result.group(1))
            # print(x)
            filepath = os.path.join(folder,file.name)
            print(file.name)
            match data_type:
                case "freeEnergy": data = getFreeEnergyFromFile(filepath)[0]
                case "hirshfeld" : 
                    if len(comp_dict) > 0:
                       for key,value in comp_dict.items():
                           if (key == result.group(1)):
                               data = getHirshFromFile(filepath, value[a]+1)
                
                case "lumo" : data = getFMOFromFile(filepath, fmo_type="lumo")
                case "lumoNPO" :  
                    table = getNPOFMOTableFromFile(file,fmo="lumo").transpose()
                    data = table[table.columns[-1]].min()                
                case "homo" : data = getFMOFromFile(filepath, fmo_type="homo")
                case "homoNPO" : 
                    table = getNPOFMOTableFromFile(file,fmo="homo").transpose()
                    data = table[table.columns[-1]].max()
                case "dist" : 
                    if len(comp_dict) > 0:
                        for key,value in comp_dict.items():
                          if (key in result.group(1)):  
                              atom_list = [value[0]+1,value[1]+1]
                              print(atom_list)                              
                              d = getDistSumFromFile(filepath,atom_list, limit = 2.3)
                              if len(d) > 1 :
                                  data = d[-2]
                              else:
                                  data =0 
                              
                case _: data = 0
            # if (data != 0):
            y.append(data)  
            # print(y)
    return createTable(x, y)    


####   MAIN BLOCK   #######################


if __name__ == "__main__":
    
    path = os.path.join(os.getcwd(),"f")
   
    full_filename = os.path.join(os.getcwd(),"E:\\3_QM\\Reactionsearch_QM\\1_LUMO_acid_effect\\e1.log")
    files = [full_filename]
        
    fields = ["summary","bonds","energies","frequencies","nbo_summary","nbo_fock","nho_directionality",
              "nbo_start","npa_summary","nao_occupancies","mull_charges",
              "ao_scf","pa_scf","termination"]
    
    # e_dict_full = {'e1': [12, 21], 'e1043': [13, 14], 'e1096': [14, 23], 'e1218': [12, 24], 'e1269': [9, 28], # first value - main atom, second - last atom
    #           'e1295': [13, 26], 'e151': [2, 30], 'e1578': [3, 26], 'e2': [12, 25], 'e227': [12, 25], 
    #           'e257': [6, 21], 'e3': [21, 22], 'e347': [18, 43], 'e4': [17, 26], 'e416': [26, 44], 
    #           'e5': [17, 29], 'e52': [1, 22], 'e520': [12, 27], 'e523': [1, 20], 'e541': [12, 27], 'e7': [11,25],
    #           'e8': [14, 26], 'e846': [10, 28], 'e894': [10, 31], 'e9': [12, 13], 'e916': [14, 18], 
    #           'e951': [12, 30], 'e965': [9, 25]}
    
    e_dict = {'e1': [12, 21, 11], 'e1043': [13, 14], 'e1096': [14, 23], 'e1218': [12, 24], 'e1269': [9, 28], # first value - main atom, second - last atom
              'e1295': [13, 26], 'e151': [2, 30], 'e2': [12, 25, 9], 'e227': [12, 25], 
              'e257': [6, 21], 'e3': [21, 22, 13], 'e347': [18, 43], 'e4': [17, 26, 10], 'e416': [26, 44], 
              'e5': [17, 29, 20], 'e52': [1, 22], 'e520': [12, 27], 'e523': [1, 20], 'e541': [12, 27], 'e7': [11,25, 12],
              'e8': [14, 26, 5], 'e846': [10, 28], 'e894': [10, 31], 'e9': [12, 13, 11], 'e916': [14, 18], 
              'e951': [12, 30]}
    
    nu_dict = {'nu1' : [16,21,8], "nu2" : [22,22,10], "nu3" : [26,26,13], "nu4" : [28,28,0], "nu5" : [17,18,11], "nu8" : [19,19,17], "nu9" : [23,23,7]}
    
    e_small = {'e1': [12, 21,11], 'e2': [12, 25, 9], 'e3': [21, 22, 13], 'e4': [18, 26, 10], 'e5': [17, 29, 20], 'e7': [11,26, 12], 'e8': [1, 26, 5], 'e9': [12, 13, 11]}
    e_new = {"1_e1": [1,22,0],"1_e2" : [12,29,10],"1_e3" : [12,25,11],"1_e4" : [12,30,11],"1_e5" : [4,26,3], "1_e6" : [12,20,11], "1_e7" : [12,21,10], 
             "1_e8" : [17,36,14], "1_e9" : [11,26,13], "1_e10" : [11,26,13], "1_e11" : [8,19,7], "1_e12" : [12,22,3], "1_e13" : [2,18,8],
             "1_e14" : [2,25,3], "1_e15" : [2,11,1]}
    
    
    acid_dict_bonds = {'ac' : [7,5], 'for' : [4,2], 'po' : [5,3], 'hf' : [0,1], 'hpo' : [3,2], 'tf' : [7,6]} # first value - main atom, scond value - oxygen or F
     
    {'ms' : [8,7], 'hcl' : [1,0]}
    
    acid_hf_tf = {'hf' : [0,1], 'tf' : [7,6]}
       
    acid_dict = {'no' : [], 'ac' : [7,5], 'for' : [4,2], 'po' : [5,3], 'hf' : [1,0], 'hpo' : [3,2], 'tf' : [7,6],  "ms": [8,7], "hcl" : [1,0],'h' : [0]} # first value - main atom, scond value - oxygen or F
    acid_n_h = {'no' : [], 'ac' : [7,5], 'for' : [4,2],  'hf' : [1,0], 'hpo' : [3,2], 'tf' : [7,6],  "ms": [8,7], "hcl" : [1,0]} # first value - main atom, scond value - oxygen or F
    acid_neutral = {'no' : [], 'ac' : [7,5], 'for' : [4,2]} # first value - main atom, scond value - oxygen or F
    acid_dict_acids = {'ac' : [7,5], 'for' : [4,2], 'hf' : [1,0], 'hpo' : [3,2], 'tf' : [7,6]} # first value - main atom, scond value - oxygen or F
    acid_dict_n_hpo = {'ac' : [7,5], 'for' : [4,2], 'hf' : [1,0], 'tf' : [7,6]} # first value - main atom, scond value - oxygen or F
 
    
    e_one = {'e1': [12, 21,11]}
    acid_one = {'no' : []}
    
    e_894 = {'e894': [10, 31]}
    acid_hf = {'hf' : [1,0]}
    
    e_416 = {'e416': [26, 44]}
    acid_ac = {'ac' : [7,5]}

   
    x = [7, 4.75, 3.7, 4, 3.2, 2, 0.5, -2, -4, -10]
    x_n_h = [7, 4.75, 3.7, 3.2, 2, 0.5, -3, -7]
    x_n_hpo = [4.75, 3.7, 3.2, 0.5]
    x_hf_tf = [3.2, 0.5]
    x_neutral = [7, 4.75, 3.7]
    
    
    # a = [*range(15,36)]
    
    # a.remove(6)
    # a.append(6)
    # print(a)
    # a=[2]
    # b = [26,46]
    # b=a
     
    
    # print(getBondTable(e_dict, acid_dict,bond_type[1]))
   
    # frag, term = getFragments(filepath, "hirshfeld", times=2, contains = 'Optimized Parameters')
    # print(len(frag))
    # print(getDistFromFrag(frag[0], 22, 23))
    # print(getFMOFromFile(filepath, fmo_type="homo"))

    # hirsh =getHirshFromFile(filepath, [56], times=2, withH=True, contains='Optimized Parameters')
    # print(len(getLUMOFromFrag(frag["nbo_summary"])))
    # dist = getDistFromFile(filepath, a, times=2, contains = 'Optimized Parameters')
    # nbo_fmo = getNPOFMOTableFromFile(filepath, fmo="homo", times=2, contains = 'Optimized Parameters')
    # print(nbo_fmo)
    # nbo_fmo.to_clipboard(sep=",")
    # print(nbo_fmo.sort_values(by = nbo_fmo.sum().))
    # print(nbo_fmo.sum().sort_values(ascending=False))
    # energy = getEnergyFromFile(filepath, times=2, contains = 'Optimized Parameters')
    # freeEnergy = getFreeEnergyFromFile(filepath)
    
    # filepath = "E:\\3_QM\\Reactionsearch_QM\\4_complex\\output\\e1_nu2_po.log"
    # fmo = getFMOFromFile(filepath, times=2, contains = 'Optimized Parameters', fmo_type="lumo")
    # hirsh =getHirshFromFile(filepath, a, times=2, withH=True, contains='Optimized Parameters')
    # energy = getEnergyFromFile(filepath, times=2, contains = 'Optimized Parameters')
    # dist = getDistSumFromFile(filepath, b, limit=1.8, times=2,  contains = 'Optimized Parameters')   
    # table = createTable(dist,hirsh)
    # print(table)
    # table.to_clipboard(sep=",")
    
    folder = "E:\\3_QM\\Reactionsearch_QM\\db_2\\output"
    table = dataFromFolder(folder, "lumo", r"(e[0-9]+)_h\.log", comp_dict=e_small, a=0)
    print(table)
    table.to_clipboard(sep=",") 
    

    # folder_out = "E:\\3_QM\\Reactionsearch_QM\\8_solvent\\input\\nu"
    # folder_in = "E:\\3_QM\\Reactionsearch_QM\\6_radical\\nua_rn"
    # modifiedFilesInFolder(folder_in, folder_out, pat=r"(nu[0-9]a).log")
    # print(len(getFragments(filepath, "hirshfeld", times=2, contains = 'Optimized Parameters')[0]))
    
    
    
    # tableQuery("e_LUMO_NPO", "LUMOnpo")
    # db_path = "E:\\3_QM\\Reactionsearch_QM\\data.db"
    # appendDataToDB(db_path, "nu_LUMO_NPO", "LUMOnpo", table)
    
    # file = "E:\\3_QM\Reactionsearch_QM\\new\\1\\output\\1_e10.log"
    # table = getNPOFMOTableFromFile(file,fmo="lumo").transpose()
    # data = table[table.columns[-1]].min()
    # print(data)