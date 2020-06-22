#!/usr/bin/env python
from flask import Flask, request, render_template
from _collections import OrderedDict
from flask_cors import cross_origin
import fasttext, re, ujson as json, os, numpy as np, click
from gunicorn.app.base import BaseApplication

class cFlask(Flask) :
    def __init__(self, *args, **kargs) :
        super(cFlask, self).__init__(*args, **kargs)

    def init_parser(self, models) :
        self.config['SCHEMES'] = OrderedDict()
        self.config['LEVEL_MODELS'] = OrderedDict()
        for model in models :
            config = json.load(open(os.path.join(model, 'model.conf'), 'rt'))
            self.config['SCHEMES'][model] = config
            self.config['LEVEL_MODELS'][model] = [fasttext.load_model(os.path.join(model, fname)) for fname in config['models']]

    def info(self) :
        schemes = []
        for model_name, config in self.config['SCHEMES'].items() :
            schemes.append( dict(NAME=model_name,
                            LEVELS=config['levels'],
                            POSSIBLE_CATEGORIES=sorted([ [config['conversion'][c] for c in comb] for comb in config['combined_category'] ]),
                            MULTIPLE_LABEL=config['multi_label'],
                            SCHEME=config['scheme']))
        return schemes

    def tokenize(self, sent, lower=True) :
        if lower :
            return [w for w in re.split(r"[^a-zA-Z]+", sent.lower()) if len(w) > 1]
        else :
            return [w for w in re.split(r"[^a-zA-Z]+", sent) if len(w) > 1]

    def query(self, scheme, qry) :
        level_models = self.config['LEVEL_MODELS'][scheme]
        config = self.config['SCHEMES'][scheme]
        tokens = self.tokenize(qry)
        res = []
        for m in level_models :
            r = m.predict(' '.join(tokens), k = -1)
            res.append(dict(zip(r[0], r[1].tolist())))
        paths = []
        #for r in res :
        #    if '__label__ND/Others' in r :
        #        other_p = 0.
        #        for fld, p in r.items() :
        #            if p < r['__label__ND/Others'] :
        #                other_p += r[fld]
        #                r[fld] = 0.
        #        r['__label__ND/Others'] += other_p
        for cat in config['combined_category'] :
            path = np.array([min(r.get(k, 0. if k else 0.5), 1.) for k, r in zip(cat, res)])
            #if cat == ['__label__Environment', '__label__Plant'] :
            #    path[1] = np.power(path[1], 4)
            prob = path.prod()**(1./len(path))
            p = dict(zip(config['levels'], [c for c, p in zip(cat, path)]))
            p['Pr'] = prob
            p['Details'] = path.tolist()


            paths.append(p)
        if not config['multi_label'] :
            Pr_total = sum([ p['Pr'] for p in paths ])
            for p in paths :
                p['Pr'] /= Pr_total
            paths = sorted(paths, key=lambda p:-p['Pr'])[:1]
        else :
            paths = [p for p in paths if p['Pr'] >= 0.5]
            max_depth = {}
            for path in paths :
                depth = len([ 1 for k in config['levels'] if path[k] != '' ])
                for k in config['levels'] :
                    if path[k] != '' :
                        key = (k, path[k])
                        if key not in max_depth or max_depth[key] < depth :
                            max_depth[key] = depth
            for path in paths :
                depth = len([ 1 for k in config['levels'] if path[k] != '' ])
                m_depth = 100
                for k in config['levels'] :
                    if path[k] != '' :
                        key = (k, path[k])
                        if m_depth > max_depth[key] :
                            m_depth = max_depth[key]
                if m_depth > depth :
                    path['Pr'] = 0.
            paths = [p for p in paths if p['Pr'] >= 0.5]

        for path in paths :
            for k in config['levels'] :
                path[k] = config['conversion'].get(path[k], '')
            path['Raw'] = qry
        return sorted(paths, key=lambda x:-x['Pr'])

root_dir = os.path.dirname(os.path.abspath(__file__))
app = cFlask(__name__, template_folder=root_dir, static_folder=os.path.join(root_dir, 'static'))

@app.route("/", methods=['GET', 'POST'])
@cross_origin(allow_headers=['Content-Type'])
def index() :
    return render_template('fastSource.html')

@app.route("/schemes", methods=['GET', 'POST'])
@cross_origin(allow_headers=['Content-Type'])
def info() :
    res = app.info()
    return json.dumps(res, indent=2, sort_keys=True)


@app.route("/query", methods=['GET', 'POST'])
@cross_origin(allow_headers=['Content-Type'])
def query() :
    scheme = request.form.get('scheme', '') if request.method == 'POST' else request.args.get('scheme', '')
    qry = request.form.get('q', '') if request.method == 'POST' else request.args.get('q', '')
    paths = app.query(scheme, qry)
    return json.dumps(paths, indent=2, sort_keys=True)


@app.route("/batch_query", methods=['GET', 'POST'])
@cross_origin(allow_headers=['Content-Type'])
def batch_query() :
    result = {}
    scheme = request.form.get('scheme', '') if request.method == 'POST' else request.args.get('scheme', '')
    qry = request.form.get('q', '') if request.method == 'POST' else request.args.get('q', '')
    for q in set(qry.split('\n')) :
        res = app.query(scheme, q)
        result[q] = res
    return json.dumps(result, indent=2, sort_keys=True)


class Application(BaseApplication) :
    def __init__(self, app, options={}):
        self.options = options
        self.application = app
        super(Application, self).__init__()
    def load_config(self):
        config = { k: v for k, v in self.options.items() \
                   if k in self.cfg.settings and v is not None }
        for k, v in config.items() :
            self.cfg.set(k.lower(), v)
    def load(self):
        return self.application

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('models', nargs=-1)
@click.option('-w', '--workers', help='number of workers (processors) to be used. DEFAULT: 1', default=1)
@click.option('-b', '--bind', help='IP and port to be used for the API. DEFAULT: 0.0.0.0:4125', default='0.0.0.0:4125')
def main(models, workers, bind) :
    '''
    MODELS: folders containing the models generated by fastSrc_build
    '''
    app.init_parser(models)
    #app.query('enterobase_novel', 'enterobase')
    Application(app, dict(bind=bind, workers=workers)).run()

if __name__ == '__main__' :
    main()
