import pandas as pd
from CGRtools.files import SDFread
from CIMtools.preprocessing import Fragmentor, MetaReference
from CIMtools.preprocessing.reference import prepare_metareference

with open('DA_CGR_E_1208.sdf', encoding='cp1251') as f:
    cgrs = SDFread(f).read()
meta = ['additive.1:solvents_uniq_table.csv', 'temperature_1/K']
md = prepare_metareference(meta)
meta_data = md.fit_transform(cgrs)

fr = Fragmentor(version='2017.x', fragment_type=9, min_length=1, max_length=5, useformalcharge=1)
dsc = fr.fit_transform(cgrs)
dsc = pd.concat([dsc, meta_data], axis=1)
dsc.to_csv('DA_frags_E_1208.csv', sep=';', index=False)
