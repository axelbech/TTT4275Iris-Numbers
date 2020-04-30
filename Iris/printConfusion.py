import numpy as np

def print_confusion(conf):
    """
    Creates a latex table that is pre-formatted for a given confusion matrix.
    """
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


    print("""\\begin{table}[H]
\\caption{}
\\centering
\\begin{tabular}{|c|lll|}""")
    conf = conf.astype(int)
    print('\\hline\nclass & '+' & '.join(classes) + '\\\\' + '\\hline')
    for i, row in enumerate(conf):
        rw = classes[i]
        for j, elem in enumerate(row):
            rw += ' & '
            if elem == 0:
                rw += '-'
            else:
                rw += str(elem)
        rw += '\\\\'
        if i == 2:
            rw += '\\hline'
        print(rw)
    print("""\\end{tabular}
\\end{table}""")
    print()