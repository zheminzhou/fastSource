#!/usr/bin/env python
import os, sys, numpy as np, pandas as pd, re, subprocess
import ujson as json
from collections import defaultdict
import fasttext
import click, logging
from multiprocessing import Pool

FORMAT = '%(asctime)-15s | %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def tokenize(sent, lower=True) :
    if lower :
        return [w for w in re.split(r"[^a-zA-Z]+", sent.lower()) if len(w) > 1]
    else :
        return [w for w in re.split(r"[^a-zA-Z]+", sent) if len(w) > 1]


def separate_data(data, test_p, repeat) :
    np.random.seed(44)
    n_data = data.shape[0]
    p = test_p/100. if test_p <= 50. else 1. - (test_p/100.)
    acc = 0.
    separations = []
    for ite in np.arange(repeat) :
        n_acc = np.round(acc + n_data * p, 5)
        if np.ceil(acc/n_data) != np.ceil(n_acc/n_data) :
            ids = np.random.permutation(n_data)
        nt = (np.round(n_acc) - np.round(acc)).astype(int)
        acc = n_acc
        test, train = ids[:nt], ids[nt:]
        ids = np.concatenate([ids[nt:], ids[:nt]])
        if test_p > 50. :
            test, train = train, test
        separations.append([train, test])
    return separations

def ite_repeat(data) :
    ite, rep, output, levels, benchmarks, pretrained, multi_label = data

    train_set, test_set = benchmarks[rep[0]], benchmarks[rep[1]]
    res = [ [[]] + [{} for l in levels ] for d in benchmarks]
    for i, d in zip(rep[1], test_set) :
        res[i][0] = d.tolist()

    for id, (level, _) in enumerate(levels) :
        train = '{0}/train.{1}.{2}'.format(output, ite, level.replace(' ', '_'))
        with open(train, 'w') as fout :
            for d in train_set :
                fout.write('{0} {1}\n'.format( d[id+2], d[0] ))
        with open('{0}/test.{1}.{2}'.format(output, ite, level.replace(' ', '_')), 'w') as fout :
            for d in test_set :
                fout.write('{0} {1}\n'.format( d[id+2], d[0] ))
        if multi_label :
            model = fasttext.train_supervised(input=train, lr=0.2, epoch=20, wordNgrams=2, loss='ova',\
                                              bucket=200000, dim=300, verbose=0, pretrainedVectors=pretrained)
        else :
            model = fasttext.train_supervised(input=train, lr=0.2, epoch=20, wordNgrams=2,\
                                              bucket=200000, dim=300, verbose=0, pretrainedVectors=pretrained)
            
        pred, prob = model.predict(test_set.T[0].tolist(), k=-1)
        for i, t, p, d in zip(rep[1], pred, prob, test_set) :
            res[i][id+1] = dict(zip(t, p.tolist()))
    with open('{0}/fastSrc.{1}.json'.format(output, ite), 'w') as fout :
        fout.write(json.dumps( res ))
    return ite, '{0}/fastSrc.{1}.json'.format(output, ite)


def runRepeats(output, levels, benchmarks, repeats, pretrained, multi_label, combinations, conversion) :
    result = [[0]+[defaultdict(float) for l in levels ] for d in benchmarks]
    precisions, sensitivities, f1_scores, perLabel = [], [], [], defaultdict(lambda : np.zeros([3], dtype=float))
    with open('{0}/batch_eval.txt'.format(output), 'w') as fout :
        for ite, res in pool.imap_unordered(ite_repeat, [ \
                [ite, rep, output, levels, benchmarks, pretrained, multi_label] for ite, rep in enumerate(repeats) ]) :
            with open(res, 'r') as fin :
                res = json.loads(fin.read())
            precision, sensitivity, f1_score, pL = getAccuracies(res, levels, combinations, multi_label)
            logger.info('Unit {1} - Precision:\t{0}'.format('\t'.join( \
                ['{0}: {1:.3f}'.format(lvl, v) for (lvl, _), v in zip(levels + [['Overall', 0]], precision)]), ite))
            logger.info('Unit {1} - Sensitivity:\t{0}'.format('\t'.join( \
                ['{0}: {1:.3f}'.format(lvl, v) for (lvl, _), v in zip(levels + [['Overall', 0]], sensitivity)]), ite))
            logger.info('Unit {1} - F1_score:\t{0}'.format('\t'.join( \
                ['{0}: {1:.3f}'.format(lvl, v) for (lvl, _), v in zip(levels + [['Overall', 0]], f1_score)]), ite))
            fout.write('Unit {1} - Precision:\t{0}\n'.format('\t'.join( \
                ['{0}: {1:.3f}'.format(lvl, v) for (lvl, _), v in zip(levels + [['Overall', 0]], precision)]), ite))
            fout.write('Unit {1} - Sensitivity:\t{0}\n'.format('\t'.join( \
                ['{0}: {1:.3f}'.format(lvl, v) for (lvl, _), v in zip(levels + [['Overall', 0]], sensitivity)]), ite))
            fout.write('Unit {1} - F1_score:\t{0}\n'.format('\t'.join( \
                ['{0}: {1:.3f}'.format(lvl, v) for (lvl, _), v in zip(levels + [['Overall', 0]], f1_score)]), ite))

        precisions.append(precision)
        sensitivities.append(sensitivity)
        f1_scores.append(f1_score)
        for k, v in pL.items() :
            perLabel[k] += v
        for r, r0 in zip(res, result) :
            if len(r[0]) > 0 :
                r0[0] += 1
                for rr0, rr in zip(r0[1:], r[1:]) :
                    for k, v in rr.items() :
                        rr0[k] += v
    with open('{0}/summary_stats.txt'.format(output), 'w') as fout :
        for tag, stat in zip(['Precision', 'Sensitivity', 'F1_score'], [precisions, sensitivities, f1_scores]) :
            stat = np.array(stat)
            mean, min, max = np.mean(stat, 0), np.min(stat, 0), np.max(stat, 0)
            for (lvl, _), a, m, x in zip(levels + [['Overall', -1]], mean, min, max) :
                logger.info('Summary - {0}\t{1}\t{2:.5f} ({3:.5f} - {4:.5f})'.format(tag, lvl, a, m, x ))
                fout.write('Summary - {0}\t{1}\t{2:.5f} ({3:.5f} - {4:.5f})\n'.format(tag, lvl, a, m, x))
        logger.info('Summaries are also saved in {0}/summary_stats.txt'.format(output))


    with open('{0}/category_breakdown.txt'.format(output), 'w') as fout :
        fout.write('#Level\t#Category\t|\tTrue_positive\tFalse_positive\tFalse_negative\n')
        for label, stat in sorted(perLabel.items()) :
            fout.write('{0}\t{1}\t|\t{2}\t{3}\t{4}\n'.format(label[0], label[1], stat[0], stat[1], stat[2]))
        logger.info('Breakdowns for each category are saved in {0}/category_breakdown.txt'.format(output))

    for r in result :
        r[1:] = [ { k:v/r[0] for k, v in rr.items() } for rr in r[1:] ]

    back_conv = { v:k for k, v in conversion.items() }
    with open('{0}/failed_predictions.txt'.format(output), 'w') as fout :
        logger.info('Failed predictions are saved in {0}/failed_predictions.txt'.format(output))
        fout.write('RAW\t|\tTRUTH\t|\tPREDICTION\n')
        for data, res in zip(benchmarks, result) :
            pred = getPrediction(res[1:], combinations, multi_label)
            truth = [set(d.split(' ')) - set(['']) for d in data[2:]]
            t = json.dumps([sorted(x) for x in truth], sort_keys=True)
            p = json.dumps([sorted(x) for x in pred], sort_keys=True)
            if t != p :
                fout.write('{0}\t|\t{1}\t|\t{2}\n'.format(data[1], '\t'.join([','.join(sorted([back_conv[tt] for tt in t])) for t in truth]), \
                                                          '\t'.join([','.join(sorted([back_conv[tt] for tt in t])) for t in pred])))
    return result

def geomean(arr) :
    return np.multiply.reduce(arr)**(1.0/len(arr))

def getPrediction(res, combinations, multi_label) :
    if multi_label:
        paths = [ [geomean([rr.get(c, 0.) if c != '' else .5 for c, rr in zip(comb, res)]), comb] for comb in combinations ]
        paths = np.array([path[1] for path in paths if path[0] >= 0.5])
        pred = [set(np.unique(p).tolist()) - set(['']) for p in paths.T] if len(paths) else [set([])] * len(res)
    else :
        paths = [[geomean([rr.get(c, 0.) if c != '' else .5 for c, rr in zip(comb, res)]), comb] for comb in
                 combinations]
        pred = [set([p]) for p in np.array(max(paths)[1])]
    return pred


def getAccuracies(res, levels, combinations, multi_label) :
    y_true, y_pred1 = [], []
    for r in res :
        if len(r[0]) > 0 :
            y_true.append([set(d.split(' ')) - set(['']) for d in r[0][2:]])
            yp1 = getPrediction(r[1:], combinations, multi_label)
            y_pred1.append(yp1)

    perLabel = defaultdict(lambda : np.zeros(3, dtype=float))
    stats = np.zeros([3, len(levels)+1], dtype=float)
    for t, p1 in zip(y_true, y_pred1) :
        for i, ((lvl, _), truth, pp1) in enumerate(zip(levels, t, p1)) :
            hits = (truth & pp1)
            miss = truth - hits
            false = pp1 - hits
            stats[:, i] += [len(hits), len(miss), len(false)]

            for cId, group in enumerate([hits, miss, false]) :
                for c in group :
                    key = (lvl, c)
                    perLabel[key][cId] += 1
        truth = json.dumps([sorted(x) for x in t],sort_keys=True)
        pp1 = json.dumps([sorted(x) for x in p1],sort_keys=True)
        stats[:, len(levels)] += [truth == pp1, truth != pp1, truth != pp1]

    precisions = stats[0, :]/(stats[2, :]+stats[0, :])
    sensitivities = stats[0, :]/(stats[1, :]+stats[0, :])
    f1_scores = 2./(1./precisions + 1./sensitivities)

    return precisions, sensitivities, f1_scores, perLabel
    
def ite_gModel(data) :
    id, level, output, train_set, pretrained, multi_label = data
    train = '{0}/model.{1}.train'.format(output, level.replace(' ', '_'))
    with open(train, 'w') as fout :
        for d in train_set :
            fout.write('{0} {1}\n'.format(d[id+2], d[0]))
    if multi_label :
        model = fasttext.train_supervised(input=train, lr=0.2, epoch=20, wordNgrams=2, loss='ova', \
                                          bucket=200000, dim=300, pretrainedVectors=pretrained, verbose=0)
    else :
        model = fasttext.train_supervised(input=train, lr=0.2, epoch=20, wordNgrams=2, \
                                          bucket=200000, dim=300, pretrainedVectors=pretrained, verbose=0)

    model.save_model('{0}/model.{1}.bin'.format(output, level.replace(' ', '_')))
    return id, '{0}/model.{1}.bin'.format(output, level.replace(' ', '_'))

def generateModel(output, levels, benchmark, pretrained, multi_label, conversion, combined_category) :
    train_set = benchmark
    back_conv = {v:k for k, v in conversion.items()}
    configuration = dict(
        levels = [ l for l, _ in levels ],
        conversion = back_conv,
        combined_category = list(combined_category),
        models = ['' for l in levels],
        multi_label=multi_label,
    )
    for id, model_file in pool.imap_unordered(ite_gModel, [ [id, level, output, train_set, pretrained, multi_label] for id, (level, _) in enumerate(levels) ]) :
        configuration['models'][id] = os.path.basename(model_file)
    json.dump(configuration, open(output+'/model.conf', 'w'))
    logger.info('Model constructed. Use fastSource {0} to start a web API with the model'.format(os.path.abspath(output)))


def getPretrained() :
    home_dir = os.path.dirname(os.path.abspath(__file__))
    vec_file = os.path.join(home_dir, 'wiki-2016.en.vec')
    if not os.path.isfile(vec_file) :
        logger.info('wiki-2016.en.vec is not present. Start to download. This will take a while...')
        uri = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec'
        subprocess.Popen('curl -o {0} {1}'.format(vec_file, uri).split(), stdout=subprocess.PIPE).wait()
        logger.info('wiki-2016.en.vec is downloaded.')
    return vec_file

def getBenchmark(data_file, conversion, levels, combined_category, multi_label) :
    data = pd.read_csv(data_file, sep='\t', header=0, na_filter=False, encoding='iso-8859-1', dtype=str)
    cols = data.columns.tolist()
    try :
        level2 = [ [level, cols.index(level)] for level, id in levels ]
    except :
        raise ValueError('Inconsistent names of categories between definition file and dataset file')

    raw_level = [id for id, lvl in enumerate(cols) if lvl.upper().startswith('RAW')][0]
    benchmark = {}
    back_conv = {v:k for k, v in conversion.items()}

    for data_i, d in enumerate(data.values) :
        comb = []
        for level, id in level2 :
            if d[id] == '' and not multi_label :
                d[id] = '__'.join([c.replace('__label__', '') for c in comb]) + '__(ND/Others)'
            comb.append(' '.join([ conversion[dd] for dd in d[id].split(',')]))
        comb = tuple(comb)
        if not multi_label and comb not in combined_category :
            raise ValueError('Combination of {0} is not present in the definition file'.format(str(comb)))
        raw = ' '.join(tokenize(d[raw_level]))
        if raw in benchmark:
            if benchmark[raw][1] != comb :
                logger.warning('Conflicting assignments between two similar or identical records:\n{0}\n{1}\n\n'.format(
                    '{0}\t|\t{1}'.format(benchmark[raw][0], '\t|\t'.join([ ','.join(back_conv[f] for f in fld.split(' ')) for fld in benchmark[raw][1]  ])),
                    '{0}\t|\t{1}'.format(d[raw_level], '\t|\t'.join([','.join(back_conv[f] for f in fld.split(' ')) for fld in comb  ])),
                ))
        else :
            benchmark[raw] = [d[raw_level], comb]

    benchmark = np.array([[k, v[0]] + list(v[1]) for k, v in sorted(benchmark.items())])
    logger.info('Retrieved {0} non-redundant records from {1} samples.'.format(len(benchmark), len(data.values)))
    return benchmark

def prepare_data(def_file, data_file, multi_label) :
    level_def = pd.read_csv(def_file, sep='\t', header=0, na_filter=False, encoding='iso-8859-1', dtype=str)
    level_col = level_def.columns.astype(str)
    levels = [[lvl, id] for id, lvl in enumerate(level_col)]

    conversion = {}
    combined_category = []
    for d in level_def.values:
        comb = []
        for level, id in levels:
            if d[id] == '' and not multi_label:
                d[id] = '__'.join([c.replace('__label__', '') for c in comb]) + '__(ND/Others)'
            if d[id] not in conversion:
                conversion[d[id]] = '__label__{0}'.format(d[id].replace(' ', '_')) if d[id] else ''
            comb.append(conversion[d[id]])
        combined_category.append(comb)
    combined_category = {tuple(c) for c in combined_category}
    logger.info('Loaded in {0} unique combinations in {1} levels.'.format(len(combined_category), len(levels)))
    benchmark = getBenchmark(data_file, conversion, levels, combined_category, multi_label)
    return levels, combined_category, benchmark, conversion

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-d', '--definition', required=True, help='[REQUIRED] A definition file for the scheme')
@click.option('-s', '--samples', required=True, help='[REQUIRED] A set of manually curated entries for the training of the model.')
@click.option('-o', '--output', required=True, help='[DEFAULT: fastSource] the prefix for the generated models.')
@click.option('--test_p', default=20, help='[DEFAULT: 20] Percentage of samples in the test set. The others are used for the training. Set to 0 to disable the evaluation stage.')
@click.option('--repeat', default=10, help='[DEFAULT: 10] Number of independent evaluations. Use <test_p>*<repeat> >= 100 to test throught the all dataset.')
@click.option('--pretrained', default='', help='file pointer to a pre-trained vector. will run without a pre-trained vector is not flagged Use "wiki-2016" to download and use a vector based on wiki 2016.')
@click.option('--multi_label', default=False, is_flag=True, help='[DEFAULT: False] Flag this to allow multiple labels.')
@click.option('-n', '--n_proc', default=10, help='[DEFAULT: 10] Number of processors to be used.')
def main(output, definition, samples, test_p, repeat, pretrained, multi_label, n_proc) :
    global pool
    pool = Pool(n_proc)
    if pretrained.lower() == 'wiki-2016' :
        pretrained = getPretrained()

    logger.info('COMMAND: {0}'.format(' '.join(sys.argv)))
    levels, combined_category, benchmark, conversion = prepare_data(definition, samples, multi_label)
    try :
        os.makedirs(output)
    except :
        pass
    if test_p > 0 :
        repeats = separate_data(benchmark, test_p, repeat)
        runRepeats(output, levels, benchmark, repeats, pretrained, multi_label, combined_category, conversion)
    generateModel(output, levels, benchmark, pretrained, multi_label, conversion, combined_category)


pool = None
if __name__ == '__main__' :
    main()
