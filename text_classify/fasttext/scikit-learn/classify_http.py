from sanic import Sanic
from sanic.response import json
import pickle
import re
import jieba
import json as JSON
from sanic import response

app = Sanic()
models = None
test_classifiers = None
digital_pat = None


def get_models(models_file):
    return pickle.load(open(models_file, 'rb'))


def get_pat():
    return re.compile('\d+')


@app.listener('after_server_start')
async def notify_server_started(app, loop):
    global models
    global test_classifiers
    global digital_pat
    models_file = r'/data/code/github/machine_learn_msw/text_classify/fasttext/scikit-learn/models'
    models = get_models(models_file)
    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'GBDT']
    digital_pat = get_pat()
    print('Server successfully started!')


@app.route("/")
async def hello(request):
    return json({"hello": "world"})


@app.route("/get_label")
async def get_label(request):
    return json({'received': True,'request': request.json})


def result(code, message, data):
    return {'code':code, 'message':message, 'data':data}


def jsonp_result(json_result, callback):
    result_str = JSON.dumps(json_result)
    return callback + "(" + result_str + ")"


def parse_text(text):
    text = re.sub(digital_pat, '', text)
    return text


def gen_feature(parsed_text):
    word_list = jieba.lcut(parsed_text)
    text_features = dict([(w, True) for w in word_list])
    return text_features


def model_predict(text_features):
    pred_res = {}
    for name, model in models.items():
        print(name)
        p1 = model.classify(text_features)
        #p2 = model.prob_classify(test_classifiers)
        print(name + ":" + p1)
        pred_res[name] = p1

    return pred_res


def pred_res_count(pred_res):
    labels_count = {}
    for key, value in pred_res.items():
        if key in labels_count.keys():
            labels_count[value] = labels_count[value] + 1
        else:
            labels_count[value] = 1
    return labels_count


@app.route("/classify")
async def classify(request):
    args = request.args
    keys = request.args.keys()
    if 'text' not in keys:
        return result(1, 'there has no text', None)
    callback = None
    if 'callback' in keys:
        callback = args['callback'][0]

    parsed_text = parse_text(args['text'][0])
    text_features = gen_feature(parsed_text)
    label_data = model_predict(text_features)
    #label_count = pred_res_count(label_data)
    #return result(0, 'success', {'label_data':label_data, 'label_count':label_count})
    result_map = result(0, 'success', {'text':args['text'][0], 'label_data': label_data})
    if callback:
        return response.text(jsonp_result(result_map, callback))
    else:
        return response.json(result_map)


@app.route('/parse')
async def parse(request):
    res_map = {}
    for key, value in request.args.items():
        res_map[key] = value
    res_map['parsed'] = True
    res_map['url'] = request
    res_map['query_string'] = request.query_string
    return json(res_map)


@app.route('/file')
async def get_file(request):
    if 'name' in request.args.keys():
        return await response.file('/data/code/github/machine_learn_msw/text_classify/fasttext/scikit-learn/html_file.html')
    else:
        return response.json(result(1, 'error', None))


@app.route("/query_string")
def query_string(request):
    return json({ "parsed": True, "args": request.args, "url": request.url, "query_string": request.query_string})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)