import traceback
from util.timer import Timer
from util.metrics import accuracy
from Naive_Bayes import NaiveBayes
from Perceptron import Perceptron
from Regression import LogisticRegression
from Data_Prep import MNISTData_Prep
from HyperParameters import HyperParametersAndTransforms


def get_name(obj):
    try:
        if hasattr(obj, '__name__'):
            return obj.__name__
        else:
            return obj
    except Exception as e:
        return obj


def catch_throw(e, err):
    trace = traceback.format_exc()
    print(err + f"\n{trace}")
    raise e


class RunModel():
    t1 = '\t'
    t2 = '\t\t'
    t3 = '\t\t\t'

    def __init__(self, model, model_params):
        self.model_name = model.__name__
        self.model_params = model_params
        self.model = self.build_model(model, model_params)

    def build_model(self, model, model_params):
        print("="*50)
        print(f"Building model {self.model_name}")

        try:
            model = model(**model_params)
        except Exception as e:
            err = f"Exception caught while building model for {self.model_name}:"
            catch_throw(e, err)
        return model

    def fit(self, *args, **kwargs):
        print(f"Training {self.model_name}...")
        print(f"{self.t1}Using hyperparameters: ")
        [print(f"{self.t2}{n} = {get_name(v)}")
         for n, v in self.model_params.items()]
        try:
            return self._fit(*args, **kwargs)
        except Exception as e:
            err = f"Exception caught while training model for {self.model_name}:"
            catch_throw(e, err)

    def _fit(self, X, y, metrics=None, pass_y=False):
        if pass_y:
            self.model.fit(X, y)
        else:
            self.model.fit(X)
        preds = self.model.predict(X)
        scores = self.get_metrics(y, preds, metrics, prefix='Train')
        return scores

    def evaluate(self, *args, **kwargs):
        print(f"Evaluating {self.model_name}...")
        try:
            return self._evaluate(*args, **kwargs)
        except Exception as e:
            err = f"Exception caught while evaluating model for {self.model_name}:"
            catch_throw(e, err)

    def _evaluate(self, X, y, metrics, prefix=''):
        preds = self.model.predict(X)
        scores = self.get_metrics(y, preds, metrics, prefix)
        return scores

    def predict(self, X):
        try:
            preds = self.model.predict(X)
        except Exception as e:
            err = f"Exception caught while making predictions for model {self.model_name}:"
            catch_throw(e, err)

        return preds

    def get_metrics(self, y, y_hat, metrics, prefix=''):
        scores = {}
        for name, metric in metrics.items():
            score = metric(y, y_hat)
            display_score = round(score, 3)
            scores[name] = score
            print(f"{self.t2}{prefix} {name}: {display_score}")
        return scores


def run_eval(eval_stage='validation'):
    main_timer = Timer()
    main_timer.start()

    total_points = 0

    task_info = [
        dict(
            model=Perceptron,
            name='Perceptron',
            data=MNISTData_Prep,
            data_prep=dict(binarize=True, return_array=True),
            metrics=dict(acc=accuracy),
            eval_metric='acc',
            trn_score=0,
            eval_score=0,
            successful=False,
        ),
        dict(
            model=NaiveBayes,
            name='NaiveBayes',
            data=MNISTData_Prep,
            data_prep=dict(return_array=True),
            metrics=dict(acc=accuracy),
            eval_metric='acc',
            trn_score=0,
            eval_score=0,
            successful=False,
        ),
        dict(
            model=LogisticRegression,
            name='LogisticRegression',
            data=MNISTData_Prep,
            data_prep=dict(return_array=True),
            metrics=dict(acc=accuracy),
            eval_metric='acc',
            trn_score=0,
            eval_score=0,
            successful=False,
        ),
    ]

    total_points = 0

    for info in task_info:
        task_timer = Timer()
        task_timer.start()
        try:
            params = HyperParametersAndTransforms.get_params(info['name'])
            model_kwargs = params.get('model_kwargs', {})
            data_prep_kwargs = params.get('data_prep_kwargs', {})

            run_model = RunModel(info['model'], model_kwargs)
            data = info['data'](**data_prep_kwargs)
            X_trn, y_trn, X_vld, y_vld = data.data_prep(**info['data_prep'])

            trn_scores = run_model.fit(
                X_trn, y_trn, info['metrics'], pass_y=True)
            eval_scores = run_model.evaluate(
                X_vld, y_vld, info['metrics'], prefix=eval_stage.capitalize())

            info['trn_score'] = trn_scores[info['eval_metric']]
            info['eval_score'] = eval_scores[info['eval_metric']]
            info['successful'] = True

        except Exception as e:
            track = traceback.format_exc()
            print(
                "The following exception occurred while executing\n", track)
        task_timer.stop()

    print("="*50)
    print('')
    main_timer.stop()

    avg_trn_acc, avg_eval_acc, successful_tests = summary(task_info)
    task_eval_acc = get_eval_scores(task_info)

    print(f"Average Train Accuracy: {avg_trn_acc}")
    print(f"Average {eval_stage.capitalize()} Accuracy: {avg_eval_acc}")

    return (total_points, avg_eval_acc, main_timer.last_elapsed_time, avg_trn_acc, *task_eval_acc)


def summary(task_info):
    sum_trn_acc = 0
    sum_eval_acc = 0
    successful_tests = 0

    for info in task_info:
        if info['successful']:
            successful_tests += 1
            sum_trn_acc += info['trn_score']
            sum_eval_acc += info['eval_score']

    if successful_tests == 0:
        return 0, 0, successful_tests

    avg_trn_acc = sum_trn_acc / len(task_info)
    avg_eval_acc = sum_eval_acc / len(task_info)
    return round(avg_trn_acc, 4), round(avg_eval_acc, 4), successful_tests


def get_eval_scores(task_info):
    return [i['eval_score'] for i in task_info]


if __name__ == "__main__":
    run_eval()
