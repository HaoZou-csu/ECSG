# -*- coding: utf-8 -*-

'''
@Time    : 2021/4/20 13:57
@Author  : Zou Hao

'''
import os
import pandas as pd
import pymatgen as mat
from pymatgen.core import Composition
from itertools import product
import numpy as np
import smact
from smact import neutral_ratios
from smact.screening import pauling_test
from smact.screening import pauling_test_old
import time



# formula = 'MgNaClO8'
abs_file_path = os.path.dirname(__file__)

def check_neutrality(formula):
    comp = Composition(formula)
    reduce_formula = comp.get_el_amt_dict()
    element_num = len(list(reduce_formula.keys()))
    max_num = max(reduce_formula.values())
    element_list = list(reduce_formula.keys())

    elements = []
    oxidation_states = []

    stoichs = list(
        np.array(list(reduce_formula.values())).astype(np.float32)[:, np.newaxis]
    )

    # import smact
    # smact.Element('Na').oxidation_states
    ele_ox_dict = np.load(os.path.join(abs_file_path, '../data/element_ox.npy'), allow_pickle=True).item()
    for i in range(element_num - 1):
        ox = ele_ox_dict[element_list[i]]
        ox = [x for x in ox if x >0]
        oxidation_states.append(ox)
    oxidation_states.append([-2])
    for j in product(*oxidation_states):
        cn_e = np.array(j) * np.array(stoichs).ravel()
        netral_ = np.sum(cn_e)

        if netral_ == 0:
            return True, j
            # print(j)
            # break
    return False, j


def check_electronegativity(formula):
    comp = Composition(formula)
    reduce_formula = comp.get_el_amt_dict()
    element_num = len(list(reduce_formula.keys()))
    max_num = max(reduce_formula.values())
    element_list = list(reduce_formula.keys())

    elements = []
    oxidation_states = []
    pauling_electro = []

    stoichs = list(
        np.array(list(reduce_formula.values())).astype(np.int32)[:, np.newaxis]
    )

    for i in range(element_num):
        ox = smact.Element(element_list[i]).oxidation_states
        paul_elem = smact.Element(element_list[i]).pauling_eneg
        oxidation_states.append(ox)
        pauling_electro.append(paul_elem)

    if None in pauling_electro:
        print("No pauling electronegativity data")
        return False

    for j in product(*oxidation_states):
        # electroneg_makes_sense = pauling_test(j, pauling_electro, elements)
        electroneg_makes_sense = pauling_test_old(j,pauling_electro,element_list)

        if electroneg_makes_sense:
            return True
    return False


def neutrality_and_electronegativity_check(formula):
    comp = Composition(formula)
    reduce_formula = comp.get_el_amt_dict()
    element_num = len(list(reduce_formula.keys()))
    max_num = max(reduce_formula.values())
    element_list = list(reduce_formula.keys())

    elements = []
    oxidation_states = []
    pauling_electro = []

    stoichs = list(
        np.array(list(reduce_formula.values())).astype(np.int32)[:, np.newaxis]
    )

    for i in range(element_num):
        ox = smact.Element(element_list[i]).oxidation_states
        paul_elem = smact.Element(element_list[i]).pauling_eneg
        oxidation_states.append(ox)
        pauling_electro.append(paul_elem)

    if None in pauling_electro:
        print("No pauling electronegativity data")
        return False

    for j in product(*oxidation_states):
        # electroneg_makes_sense = pauling_test(j, pauling_electro, elements)
        cn_e, cn_r = neutral_ratios(j, stoichs=stoichs, threshold=int(max_num))
        electroneg_makes_sense = pauling_test_old(j, pauling_electro, element_list)
        if cn_e:
            if electroneg_makes_sense:
                return True
    return False


if __name__ == '__main__':
    tt
    start = time.time()
    data = pd.read_csv('./formula.csv')
    tu = []
    ox = []
    for i in range (len(data)):
        a,j =  check_neutrality(data['0'][i])
        if a:
            tu.append(i)
            ox.append(j)
    end = time.time()
    print(end - start)
    # print(check_neutrality('Au0.375Hg0.025Tl0.025Pb0.025Bi0.025Li0.525O'))


    # with open('huahejia_my.txt','r',encoding='utf-8') as fp:
    #     dic = []
    #     for line in fp.readlines():
    #         line=line.strip('\n') #去掉换行符\n
    #         b=line.split(':') #将每一行以空格为分隔符转换成列表
    #         b[0] = eval(b[0])
    #         b[1] = eval(b[1])
    #         dic.append(b)
    #
    # dic=dict(dic)

    data = data.iloc[tu,:]
    data.index = range(len(data))
    data.columns = ['No.','formulas']
    data['oxidation'] = ox
    data.to_csv('LiNbTmO.csv')