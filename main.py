import matplotlib.pyplot as plt
from CircuitConstruct import *
from CircuitOptimize import *
from HamiltonianGenerator import *
import os
import numpy as np 
from tqdm import tqdm


#------------------------------------------------------------------------------------------------
#                     The Function used for benchmarking a specific circuit
#                               (return a dic stores results)
#------------------------------------------------------------------------------------------------
def Benchmarking(circuit):
    #----------------------------------------------------------------------------------------  
    #                    outputCircuit after our Optimization Algorithm
    #     (Firstly using subcircuit Decomposition method to decompose k-local Pailis into 
    #              2-locals, then conduct the greedy based circuit optimization)
    #---------------------------------------------------------------------------------------- 
    local2gate = HamiltonianDecomposition(circuit)
    # store benchmarking results
    results = {}
    results['gateCount'] = {}
    results['depth'] = {}
    #----------------------------------------------------------------------------------------
    # Original Circuit
    #OriginalCircuit = ConstructCircuit(circuit)
    #OriginalCircuit = OriginalCircuit.decompose()
    # OriginalCircuit.draw('mpl')
    # get Ordered Circuit
    DFSCircuit = DFSOrdering(local2gate)
    ECCircuit = ECOrdering(local2gate)
    #---------( The following two ConstructCircuit functions are used for inspection before final output, thet're not necessary )---------
    # optimized circuits of two different strategies 
    # DFSOptimizedCircuit = ConstructCircuit(DFSCircuit)
    # DFSOptimizedCircuit.decompose().draw('mpl')
    # ECOptimizedCircuit = ConstructCircuit(ECCircuit)
    # ECOptimizedCircuit.decompose().draw('mpl')
    # optimized circuit
    #---------------------------------------------------
    Outputcircuit_DFS = Optimize(DFSCircuit,'DFS')
    Outputcircuit_DFS = Outputcircuit_DFS.decompose()
    #Outputcircuit_DFS.draw('mpl')
    Outputcircuit_EC = Optimize(ECCircuit,'EC')
    Outputcircuit_EC = Outputcircuit_EC.decompose()
    # Outputcircuit_EC.draw('mpl')
    #---------------------------------------------------------------------------------------- 
    #                   Circuit after optimization of Paulihedral Compiler
    #----------------------------------------------------------------------------------------
    paulihedral_result = PH(circuit)
    #---------------------------------------------------------------------------------------- 
    #                     Circuit after optimization of Qiskit Optimizer
    #---------------------------------------------------------------------------------------- 
    qiskit_result = Qis(circuit)
    #---------------------------------------------------------------------------------------- 
    #              Circuit after optimization of t|ket> quantum simulation programs
    #---------------------------------------------------------------------------------------- 
    tk_result = TK(circuit)
    #---------------------------------------------------------------------------------------- 
    #                    Compute Gate count and Circuit depth
    #---------------------------------------------------------------------------------------- 
    results['gateCount']['DFS'] = sum(Outputcircuit_DFS.count_ops().values())
    results['depth']['DFS'] = Outputcircuit_DFS.depth()
    results['gateCount']['EC'] = sum(Outputcircuit_EC.count_ops().values())
    results['depth']['EC'] = Outputcircuit_EC.depth()
    results['gateCount']['paulihedral'] = paulihedral_result['gateCount']
    results['depth']['paulihedral'] = paulihedral_result['depth']
    results['gateCount']['qiskit'] = qiskit_result['gateCount']
    results['depth']['qiskit'] = qiskit_result['depth']
    results['gateCount']['tket'] = tk_result['gateCount']
    results['depth']['tket'] = tk_result['depth']
    

    for optimizer in ['DFS','EC','paulihedral','qiskit','tket']:
        print('   ' + optimizer)
        print('gate count:\t' + str(results['gateCount'][optimizer]))
        print('depth:\t\t' + str(results['depth'][optimizer]) + '\n')
    
    return results

#------------------------------------------------------------------------------------------------
#          Benchmarking function for collecting results from different benchmark models
#                 (Automatically write the results to files in current fold,
#                    return a dic stores all benchmark results if needed)
#------------------------------------------------------------------------------------------------
def Benchmarks():
    BenchmarkMethods = ['DFS','EC','paulihedral','qiskit','tket']
    benchmarks_result = {}
    benchmarks_result['gateCount'] = {}
    benchmarks_result['depth'] = {}
    #--------------------output benchmarking results-------------------
    def OutputResult(folder,name, model,gateCount,depth):
        with open(folder + name + '.txt', 'w') as outputfile:
            outputfile.write('\t'+'gateCount\t'+'Depth\n')
            for (i,j,k) in zip(model,gateCount,depth):
                outputfile.write(str(i)+'\t')
                outputfile.write(str(j)+'\t\t')
                outputfile.write(str(k)+'\n')
    #--------------------------generate Hamiltonian benchmarks data---------------------------
    #----Molecule Hamiltonians----
    for method in BenchmarkMethods:
        benchmarks_result['gateCount'][method] = []
        benchmarks_result['depth'][method] = []
    molecule = tqdm(['HF','LiH','H2O','NH2','CH2','NH3','CH4'])
    for model in molecule:
        molecule.set_description("Molecule : %s" % model)
        parr = load_oplist(model, benchmark='molecule')
        result = Benchmarking(parr)
        for method in BenchmarkMethods:
            benchmarks_result['gateCount'][method].append(result['gateCount'][method])
            benchmarks_result['depth'][method].append(result['depth'][method])
    for method in BenchmarkMethods:
        OutputResult('result/molecule/molecule_', method , molecule , benchmarks_result['gateCount'][method] , benchmarks_result['depth'][method])
    #----UCCSD Hamiltonians----
    for method in BenchmarkMethods:
        benchmarks_result['gateCount'][method] = []
        benchmarks_result['depth'][method] = []
    uccsdmodel = tqdm(['LiH', 'BeH2', 'CH4'])
    for model in uccsdmodel:
        uccsdmodel.set_description("UCCSD of : %s" % model)
        parr = load_oplist(model, benchmark='uccsd')
        result = Benchmarking(parr)
        for method in BenchmarkMethods:
            benchmarks_result['gateCount'][method].append(result['gateCount'][method])
            benchmarks_result['depth'][method].append(result['depth'][method])
    for method in BenchmarkMethods:
        OutputResult('result/uccsd/uccsd_', method , uccsdmodel , benchmarks_result['gateCount'][method] , benchmarks_result['depth'][method])
    #----FermiHubbard Model Hamiltonians----
    for method in BenchmarkMethods:
        benchmarks_result['gateCount'][method] = []
        benchmarks_result['depth'][method] = []
    sizeoflattice = tqdm(range(2,6))
    fermihubbardsize = []
    for i in sizeoflattice:
        fermihubbardsize.append(str(i)+'x'+str(i))
        sizeoflattice.set_description("Fermi-Hubbard Model size of : %s" % i)
        parr = gene_FermiHubbard_oplist(i,i)
        result = Benchmarking(parr)
        for method in BenchmarkMethods:
            benchmarks_result['gateCount'][method].append(result['gateCount'][method])
            benchmarks_result['depth'][method].append(result['depth'][method])
    for method in BenchmarkMethods:
        OutputResult('result/fermihubbard/fermihubbard_', method , fermihubbardsize , benchmarks_result['gateCount'][method] , benchmarks_result['depth'][method])
    #----Random k-local Hamiltonians----
    for method in BenchmarkMethods:
        benchmarks_result['gateCount'][method] = []
        benchmarks_result['depth'][method] = []
    sizeofqubit = tqdm(range(4,13))
    randomsize = []
    for i in sizeofqubit:
        randomsize.append(str(i)+'qubits')
        sizeofqubit.set_description("Random Hamiltonians size of : %s" % i)
        parr = gene_random_oplist(i)
        result = Benchmarking(parr)
        for method in BenchmarkMethods:
            benchmarks_result['gateCount'][method].append(result['gateCount'][method])
            benchmarks_result['depth'][method].append(result['depth'][method])
    for method in BenchmarkMethods:
        OutputResult('result/random/random_', method , randomsize , benchmarks_result['gateCount'][method] , benchmarks_result['depth'][method])
    #-----------------------------------------------------------------------------------------
    


#-------------------------------------------------------------------------------------------------------
#     ⭐   Benchmarking of our algorithm and other state-of-the-art Compilers and Optimizers    ⭐
#-------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #----------------------------------------------------------------------------------------
    #              ~ Calculate benchmarking results from corresponding data ~
    #----------------------------------------------------------------------------------------
    #--------------------------------Models Benchmarking-------------------------------------
    #Benchmarks()
    #----------------------------------------------------------------------------------------
    #                ~ Plot benchmarking results from benchmark models ~
    #----------------------------------------------------------------------------------------
    # read result from files
    '''result = {}
    result['gateCount'] = {}
    result['depth'] = {}
    BenchmarkMethods = ['DFS','EC','paulihedral','qiskit','tket']
    for method in BenchmarkMethods:
        with open('result/molecule/molecule_' + method + '.txt') as file:
            filecontent = file.readlines()
            result['gateCount'][method] = []
            result['depth'][method] = []
            for line in filecontent[1:]:
                line = line.split()
                result['gateCount'][method].append(int(line[1]))
                result['depth'][method].append(int(line[2]))
    scale = range(0,7)
    plt.figure(figsize=(20,8),dpi=100)
    plt.grid(axis='y', alpha=0.75,zorder=0)
    #plt.xlabel('#qubits',fontsize=40)
    #error_params = dict(elinewidth=4,capsize=6,capthick=4)
    x_width = 0.17
    x_DFS = [i-2*x_width for i in scale]
    x_EC = [i-x_width for i in scale]
    x_paulihedral = scale
    x_qiskit = [i+x_width for i in scale]
    x_tket = [i+2*x_width for i in scale]
    labels = ['HF','LiH','H2O','NH2','CH2','NH3','CH4']
    plt.xticks(x_paulihedral,labels,fontsize=40)
    plt.yticks(fontsize=40)
    plt.bar(x_DFS,result['gateCount']['DFS'],width=x_width,label='DFS',color='lightskyblue',zorder=100)
    plt.bar(x_EC,result['gateCount']['EC'],width=x_width,label='EC',color='cornflowerblue',zorder=100)
    plt.bar(x_paulihedral,result['gateCount']['paulihedral'],width=x_width,label='Paulihedral',color='mediumblue',zorder=100)
    plt.bar(x_qiskit,result['gateCount']['qiskit'],width=x_width,label='Qiskit',color='blueviolet',zorder=100)
    plt.bar(x_tket,result['gateCount']['tket'],width=x_width,label='t|ket>',color='navy',zorder=100)
    plt.legend(fontsize=30)
    plt.ylabel('#gateCount',fontsize=40)
    plt.savefig('result/figure/molecule_gateCount.jpg',bbox_inches = 'tight')
    plt.savefig('result/figure/molecule_gateCount.eps',bbox_inches = 'tight')'''
    
    #-------------------calculate average and maximum cost reduction-------------------------
    '''result = {}
    reduction = {}
    BenchmarkMethods = ['DFS','EC','paulihedral','qiskit','tket']
    models = ['molecule','uccsd','fermihubbard','random']
    comparisons = ['paulihedral','qiskit','tket']
    items = ['gateCount','depth']
    paths = ['DFS','EC']
    result['gateCount'] = {}
    result['depth'] = {}
    for model in models:
        result['gateCount'][model] = {}
        result['depth'][model] = {}
    for method in BenchmarkMethods:
        for model in models:
            with open('result/' + model + '/' + model + '_' + method + '.txt') as file:
                filecontent = file.readlines()
                result['gateCount'][model][method] = []
                result['depth'][model][method] = []
                for line in filecontent[1:]:
                    line = line.split()
                    result['gateCount'][model][method].append(int(line[1]))
                    result['depth'][model][method].append(int(line[2]))
    #----------------------------------------------------------------------------------------
    for path in paths:
        reduction[path] = {}
        for comparison in comparisons:
            reduction[path][comparison] = {}
            for model in models:
                reduction[path][comparison][model] = {}
                for item in items:
                    reduction[path][comparison][model][item] = []
    for path in paths:
        for comparison in comparisons:
            for model in models:
                for item in items:
                    reduction[path][comparison][model][item] = np.array(result[item][model][comparison]) / np.array(result[item][model][path])
    ratio = {}
    ratio_term = {}
    for path in paths:
        ratio[path] = {}
        ratio_term[path] = {}
        for item in items:
            ratio[path][item] = {}
            ratio_term[path][item] = {}
            comparemean = []
            comparemax = []
            for comparison in comparisons:
                modelmean = []
                modelmax = []
                ratio_term[path][item][comparison] = {}
                for model in models:
                    modelmean.append(reduction[path][comparison][model][item].mean())
                    modelmax.append(reduction[path][comparison][model][item].max())
                comparemean.append(np.array(modelmean).mean())
                comparemax.append(np.array(modelmax).max())
                ratio_term[path][item][comparison]['mean'] = np.array(modelmean).mean()
                ratio_term[path][item][comparison]['max'] = np.array(modelmean).max()
            ratio[path][item]['mean'] = np.array(comparemean).mean()
            ratio[path][item]['max'] = np.array(comparemax).max()
    # print the reduction ratio of 'DFS' and 'EC' with other compilers
    print(ratio['DFS']['gateCount'])
    print(ratio['DFS']['depth'])
    print(ratio['EC']['gateCount'])
    print(ratio['EC']['depth'])
    #print(reduction['DFS']['tket']['fermihubbard']['depth'].mean())
    #print(ratio_term['EC']['depth']['tket'])
    #print(ratio_term['EC']['gateCount']['tket'])'''
    #----------------------------------------------------------------------------------------
    #                          ~ Benchmark a single circuit ~
    #     (output Order of the circuit before and after optimization, output benchmarking
    #              results, draw the circuit by plotting qiskit QuantumCircuit)
    #----------------------------------------------------------------------------------------
    #parr = gene_random_oplist(4)
    #print(parr3)
    #parr = gene_molecule_oplist(He2)
    #print(parr)
    #heisen_parr = [gene_dot_1d(29, interaction='Z')+gene_dot_1d(29, interaction='X')+gene_dot_1d(29, interaction='Y'), gene_dot_2d(4,5, interaction='Z')+gene_dot_2d(4,5, interaction='Y')+gene_dot_2d(4,5, interaction='X'), gene_dot_3d(1,2,4, interaction='Z')+gene_dot_3d(1,2,4, interaction='Y')+gene_dot_3d(1,2,4, interaction='X')]
    #print(heisen_parr[0])
    #moles = ['LiH', 'BeH2', 'CH4', 'MgH', 'LiCl', 'CO2']
    parr = gene_FermiHubbard_oplist(2,2)
    #print(fermihubbard)
    #parr = load_oplist('LiH', benchmark='uccsd')

    
    print('~ Comparing to state-of-the-art Quantum Compilers ~')
    # output benchmarking
    result = Benchmarking(parr)
    