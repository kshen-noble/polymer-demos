import numpy as np


# Generate Smiles String
def generate_diol(nO,nCO2,nCO3,nC,nCH3):
    functional_groups = []
    for (nftn,ftn) in zip( [nO,nCO2,nCO3],["O","C(=O)O","OC(=O)O"] ):
        functional_groups.extend([ftn]*nftn)

    functional_groups = np.random.permutation(functional_groups)

    nftn_total = max(1,len(functional_groups))
    nC_per = nC // nftn_total
    nC_remain = nC % nftn_total
    Cs = [nC_per*'C']*nftn_total
    for ii in range(nC_remain):
        Cs[ii] = Cs[ii]+'C'
    
    # Workaround, NEED TO UPDATE to allow for more methyl substitutions
    for ii in range(nCH3):
        Cs[ii] += "(C)"

    # Create SMILES
    smiles = ''
    for ii in range(nftn_total):
        smiles += Cs[ii]
        if len(functional_groups) > 0:
            smiles += functional_groups[ii]

    #smiles = '*'+smiles+'*'
    smiles = "O" + smiles + "O"
    return smiles


def generate_iso(nC6H6,nC6H12,nC,nmethyl,Ui=-1):
    functional_groups = []
    for (nftn,ftn) in zip( [nC6H6,nC6H12],["C6H6","C6H12"] ):
        functional_groups.extend([ftn]*nftn)

    functional_groups = np.random.permutation(functional_groups)

    if len(functional_groups) == 0:
        return "O=C=N" + nC*"C" + "N=C=O"
    
    nftn_total = max(1,len(functional_groups))

    nC_per = nC // nftn_total
    nC_remain = nC % nftn_total

    Cs = [nC_per*'C']*nftn_total
    for ii in range(nC_remain):
        Cs[ii] = Cs[ii]+'C'

    nmethyls_per = nmethyl // nftn_total
    nmethyls_remain = nmethyl % nftn_total
    methyls = [nmethyls_per*"C"]*nftn_total
    for ii in range(nmethyls_remain):
        methyls[ii]+="C"

    # figure out substitutions
    # (C's) + C[#]C{a}C{b}C{c}C{d}C{e}
    # for now assume subsequent units get attached at the c-position
    ftn_intermediates = []
    for ii in range(nftn_total):
        tmp_Cs = Cs[(ii+1)%nftn_total]
        ftn    = functional_groups[ii]
        tmp_methyls = methyls[ii]

        if ftn == "C6H6":
            tmp = "c{{idx}}{a}c{b}c{c}c{d}{{connect}}c{e}c{f}{{idx}}"
        elif ftn == "C6H12":
            tmp = "C{{idx}}{a}C{b}C{c}C{d}{{connect}}C{e}C{f}{{idx}}"
        else:
            raise ValueError("unknown ftn type {ftn}")

        attachments = ['']*6
        for ii,frag in enumerate(tmp_methyls):
            attachments[ii%6] += f"({frag})" #"(C)"
        
        tmp = tmp.format(
            b = attachments[0],
            c = attachments[1],
            e = attachments[2],
            f = attachments[3],
            a = attachments[4],
            d = attachments[5],
        )

        tmp = tmp_Cs + tmp
        ftn_intermediates.append(tmp)

    # now concatenate
    tmp = "N=C=O"
    tmp = "N-C=O" #closer to form it takes in polymer
    for ii in range(nftn_total):
        idx = nftn_total - ii
        tmp = ftn_intermediates[-1-ii].format(idx=idx,connect=f"({tmp})")

    head = {-1:"O=C=N",0:"OCCCCOC(=O)N",1:"c8c(N)c(Cl)cc(Cc9ccc(N)c(Cl)c9)c8"}[Ui]
    if Ui <= 0:
        return head + tmp
    else:
        return f"c8c(N)c(Cl)cc(Cc9ccc(N{tmp})c(Cl)c9)c8"


def DP(NCO_OH):
    return 1 / (1-1/NCO_OH)