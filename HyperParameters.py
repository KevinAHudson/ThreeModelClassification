from sklearn.preprocessing import StandardScaler
from BaseModel import ClassificationModel


class HyperParametersAndTransforms():

    @staticmethod
    def get_params(name):
        model = getattr(HyperParametersAndTransforms, name)
        params = {}
        for key, value in model.__dict__.items():
            if not key.startswith('__') and not callable(key):
                if not callable(value) and not isinstance(value, staticmethod):
                    params[key] = value
        return params

    class Perceptron():
        model_kwargs = dict(
            alpha=.01,
            epochs=3,
            seed=42
        )

        data_prep_kwargs = dict(
            target_pipe=None,
            feature_pipe=StandardScaler()

        )

    class NaiveBayes():
        model_kwargs = dict(
            smoothing=3e-2,
        )

        data_prep_kwargs = dict(
            target_pipe=None,
            feature_pipe=StandardScaler()
        )

    class LogisticRegression():
        model_kwargs = dict(
            alpha=0.01,
            epochs=140,
            seed=42,
            batch_size=64,
        )

        data_prep_kwargs = dict(
            target_pipe=None,
            feature_pipe=StandardScaler()
        )
