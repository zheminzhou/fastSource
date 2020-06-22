#!/usr/bin/env python
import click, numpy as np, pandas as pd, requests, logging, re

FORMAT = '%(asctime)-15s | %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


@click.command()
@click.option('-p', '--prefix', help='prefix for outputs', default='bacdive')
def main(prefix) :
    available_combinations = {}
    assignments = {}
    logging.info('download isolation sources')
    #with open('export_bacdive_iso_table.csv', 'w') as fout:
    #    fout.write(requests.get('https://bacdive.dsmz.de/isolation-sources?csv=1').text)
    logging.info('done')
    data = pd.read_csv('export_bacdive_iso_table.csv', header=0, na_filter=False, dtype=str)
    heading = data.columns
    raw_col = np.where(heading == 'Isolation source')[0][0]
    data = data.values
    for d in data :
        if d[raw_col] != '' :
            raw = re.sub('\s+', ' ', d[raw_col])
        labels = [ re.sub(',', ' ', l.strip('#')) for l in d[-3:] ]
        available_combinations[tuple(labels)] = 1
        if raw not in assignments :
            assignments[raw] = [ [l] for l in labels ]
        else :
            for l0, l in zip(assignments[raw], labels) :
                l0.append(l)
    with open(prefix+'_attributes.txt', 'wt') as fout :
        fout.write('Category 1\tCategory 2\tCategory 3\n')
        for comb in available_combinations.keys() :
            fout.write('\t'.join(comb)+'\n')
    with open(prefix+'_samples.txt', 'wt') as fout :
        fout.write('Raw\tCategory 1\tCategory 2\tCategory 3\n')
        for raw, assigns in sorted(assignments.items()) :
            labels = []
            for assign in assigns :
                labels.append(','.join(sorted(set(assign) - {''})))
            fout.write('{0}\t{1}\n'.format(raw, '\t'.join(labels)))
    

if __name__ == '__main__' :
    main()