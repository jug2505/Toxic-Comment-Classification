from spyne import Application, rpc, ServiceBase, Unicode
from spyne.protocol.http import HttpRpc
from spyne.protocol.json import JsonDocument
from spyne.server.wsgi import WsgiApplication
from wsgiref.simple_server import make_server
from keras.models import model_from_json
from preprocessor import Preprocessor
import numpy as np


class ClassificationService(ServiceBase):
    @rpc(Unicode, _returns=Unicode)
    def classify(ctx, sentence):
        with open("lstm_model.json", "r") as json_file:
            json_string = json_file.read()
        model = model_from_json(json_string)
        model.load_weights('lstm_weights.h5')

        test_vec = Preprocessor('corpus_marked', 'vk_comment_model') \
            .exm_pipeline(sentence, 200, 300)
        return str(np.around(model.predict(test_vec)))


app = Application([ClassificationService], tns='spyne.examples.hello.http',
                          in_protocol=HttpRpc(validator='soft'),
                          out_protocol=JsonDocument())


if __name__ == '__main__':
    wsgi_app = WsgiApplication(app)
    server = make_server('0.0.0.0', 8000, wsgi_app)
    server.serve_forever()