
from tkinter import *
from tkinter import ttk as ttk



# def _Preview(*args, **kwargs):
#     global CalcType, MethodType

def _CalcType(*args, **kwargs):
    global CalcMain, CalcTypeMain
    CalcMain = CalcTypeMain.get()

def _MethodType(*args, **kwargs):
    global MethodType
    Method  = MethodType.get()
    print(Method)
    if MethodType.get() == 'DFT':
        global FuncType

        # Creation of Functional tpype dropdown box
        Label(MethodTab, text='Functional', font = ("Times New Roman", 12)).grid(column=3, row=1, padx=10, pady=25)

        FuncType = StringVar()

def _MethodType(*args, **kwargs):
    global Method
    Method = MethodType.get()
    _MethodRemove()

    # Creation of Functional type dropdown box
    if Method == 'DFT':
        FunctionalLabel.grid(row=1, column=3, padx=10, pady=10)
        FunctionalTypes.grid(row=1 ,column=4)
        return
    if Method == 'CC':
        CCLabel.grid(row=2, column=1, padx=3, pady=10, sticky=W)
        CCSOPPALabel.grid(row=3, column=1, padx=3, pady=10, sticky=W)
        CCTypesCCS.grid(row=2, column=2, padx=3, pady=10, sticky=W)
        CCTypesMP2.grid(row=2, column=3, padx=3, pady=10, sticky=W)
        CCTypesCC2.grid(row=2, column=4, padx=3, pady=10, sticky=W)
        CCTypesCISpD.grid(row=2, column=5, padx=3, pady=10, sticky=W)
        CCTypesCCSD.grid(row=2, column=6, padx=3, pady=10, sticky=W)
        CCTypesCCSDRp3.grid(row=2, column=7, padx=3, pady=10, sticky=W)
        CCTypesCCSDpT.grid(row=2, column=8, padx=3, pady=10, sticky=W)
        CCTypesCC3.grid(row=2, column=9, padx=3, pady=10, sticky=W)
        CCTypesSOPPA.grid(row=3, column=2, padx=3, pady=10, columnspan=3)
        return

    _Preview()

def _MethodRemove(*args, **kwargs):
    if Method != 'DFT':
        FunctionalTypes.grid_remove()
        FunctionalLabel.grid_remove()

    if Method != 'CC':
        CCLabel.grid_remove()
        CCSOPPALabel.grid_remove()
        CCTypesCCS.grid_remove()
        CCTypesMP2.grid_remove()
        CCTypesCC2.grid_remove()
        CCTypesCISpD.grid_remove()
        CCTypesCCSD.grid_remove()
        CCTypesCCSDRp3.grid_remove()
        CCTypesCCSDpT.grid_remove()
        CCTypesCC3.grid_remove()
        CCTypesSOPPA.grid_remove()

def _FunctionalType(*args, **kwargs):
    global Functional
    if Method == 'DFT':
        Functional = FuncType.get()
        return


def _CCType(*args, **kwargs):
    global CC, CCSOPPA
    if Method == 'CC':
        CC = [i.get() for i in [CCTypeCCS, CCTypeMP2, CCTypeCC2, CCTypeCISpD, CCTypeCCSD, CCTypeCCSDRp3, CCTypeCCSDpT, CCTypeCC3] if i.get() != '']
        CCSOPPA = CCTypeSOPPA.get()
        return
    pass

if __name__ == '__main__':
    root = Tk()
    root.title('XYZ to DALTON20')
    tabControl = ttk.Notebook(root)

    root.geometry('800x500')

    # Tabs can be added here
    MainTab = Frame(tabControl)
    MethodTab = Frame(tabControl)

    tabControl.add(MainTab, text='Main')
    tabControl.add(MethodTab, text='Method')
    tabControl.pack(expand=1, fill="both")


    # Creation of Functional type dropdown box
    FunctionalLabel = Label(MethodTab, text='Functional', font = ("Times New Roman", 12))

    FuncType = StringVar()

    FunctionalTypes = ttk.Combobox(MethodTab, textvariable=FuncType, values=('LDA', 'BLYP', 'B3LYP','CAMB3LYP', 'B2PLYP', 'PBE', 'PBE0', 'PBE0DH'), state='readonly')
    FunctionalTypes.set('LDA')
    FunctionalTypes.bind("<<ComboboxSelected>>", _FunctionalType)
    
    # Creation of CC type radiobutton and dropdown boxex
    CCLabel = Label(MethodTab, text='CC module inputs:', font = ("Times New Roman", 12))
    CCSOPPALabel = Label(MethodTab, text='CC SOPPA inputs', font = ("Times New Roman", 12))

    CCTypeCCS = StringVar()
    CCTypeMP2 = StringVar()
    CCTypeCC2 = StringVar()
    CCTypeCISpD = StringVar()
    CCTypeCCSD = StringVar()
    CCTypeCCSDRp3 = StringVar()
    CCTypeCCSDpT = StringVar()
    CCTypeCC3 = StringVar()

    CCTypeSOPPA = StringVar()
    
    CCTypesCCS = ttk.Checkbutton(MethodTab, text='CCS', onvalue='.CCS', offvalue='', var=CCTypeCCS, command=_CCType)
    CCTypesMP2 = ttk.Checkbutton(MethodTab, text='MP2', onvalue='.MP2', offvalue='', var=CCTypeMP2, command=_CCType)
    CCTypesCC2 = ttk.Checkbutton(MethodTab, text='CC2', onvalue='.CC2', offvalue='', var=CCTypeCC2, command=_CCType)
    CCTypesCISpD = ttk.Checkbutton(MethodTab, text='CCS(D)', onvalue='.CCS(D)', offvalue='', var=CCTypeCISpD, command=_CCType)
    CCTypesCCSD = ttk.Checkbutton(MethodTab, text='CCSD', onvalue='.CCSD', offvalue='', var=CCTypeCCSD, command=_CCType)
    CCTypesCCSDRp3 = ttk.Checkbutton(MethodTab, text='CCSDR(3)', onvalue='.CCSDR(3)', offvalue='', var=CCTypeCCSDRp3, command=_CCType)
    CCTypesCCSDpT = ttk.Checkbutton(MethodTab, text='CCSD(T)', onvalue='.CCS(T)', offvalue='', var=CCTypeCCSDpT, command=_CCType)
    CCTypesCC3 = ttk.Checkbutton(MethodTab, text='CC3', onvalue='.CC3', offvalue='', var=CCTypeCC3, command=_CCType)

    CCTypesSOPPA = ttk.Combobox(MethodTab, textvariable=CCTypeSOPPA, values=['','SOPPA', 'SOPPA2', 'SOPPA(CCSD)', 'AO-SOPPA'], state='readonly')
    CCTypesSOPPA.bind("<<ComboboxSelected>>", _CCType)

    CC = []

    # Creation of Job type radiobuttons
    CalcLabel = Label(MainTab, text='Calculation Type', font = ("Times New Roman", 12))
    CalcLabel.grid(row=1, column=0, padx=10, pady=10, sticky=W)

    CalcTypeMain = StringVar()

    CalcTypesSingle = ttk.Radiobutton(MainTab, variable=CalcTypeMain, value='.RUN WAVEFUNCTION', text='Single Point', command=_CalcType)
    CalcTypesOptimize = ttk.Radiobutton(MainTab, variable=CalcTypeMain, value='.OPTIMIZE', text='Optimize', command=_CalcType)
    CalcTypeMain.set('.RUN WAVEFUNCTION')

    CalcTypesSingle.grid(row=1, column=1, padx=10, pady=10)
    CalcTypesOptimize.grid(row=1, column=2, padx=10, pady=10)

    
    # Creation of Method type dropdown box
    Label(MethodTab, text='Method', font = ("Times New Roman", 12)).grid(column=0, row=1, padx=10, pady=25)

    MethodType = StringVar()

    MethodTypes = ttk.Combobox(MethodTab, textvariable=MethodType, values=('HF', 'MP2', 'CCSD', 'DFT'), state='readonly')
    MethodTypes.current(0)
    MethodTypes.grid(row=1, column=1)
    MethodTypes.bind("<<ComboboxSelected>>", _MethodType)

    root.mainloop()