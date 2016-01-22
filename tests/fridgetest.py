from lasagnecaterer.fridge import SaveLoadZipFilemixin, OrderedDict, os, JsonSaveLoadMixin
from tempfile import mkstemp
path = os.path.abspath('tests')
fd, file = mkstemp(suffix='.zip', prefix=None, dir=path, text=False)
import atexit
from functools import partial
def closetmp():
    os.remove(file)

atexit.register(closetmp)
import numpy as np

class JSON(JsonSaveLoadMixin):
    @classmethod
    def from_dict(cls, *args, **kwargs):
        return cls(**kwargs)

    def to_dict(self) -> OrderedDict:
        return self.data

    def __init__(self, **data):
        self.data = OrderedDict(data)


class Zip(SaveLoadZipFilemixin):
    def bootstrap(self):
        pass

    @classmethod
    def from_dict(cls, *args, **kwargs):
        return cls(**kwargs)

    def to_dict(self) -> OrderedDict:
        return self.data

    def __init__(self, **data):
        self.data = OrderedDict(data)



d = {'array': np.array([1,2]),
         'listofarrays': [np.arange(10) for _ in range(10)],
         'adict': {'hej': 'med', 'foo': 'bar'},
     'zipobj': Zip(foo='bar'),
     'jsonobj': JSON(foo='bar')
         }


class TestZip:
    data = dict(d)
    testcls = Zip(**d)

    def test_save(self):
        self.testcls.save(file)

        #print('PREEEES')
        #input('press key to continue')

        new = Zip.load(file)
        for name, obj in new.data.items():
            if name == 'listofarrays':
                for it1, it2 in zip(d[name], obj):
                    assert (it1 == it2).all()
            elif name == 'zipobj' or name == 'jsonobj':
                assert obj.data['foo'] == 'bar'

            elif name == 'array':
                assert (d[name] == obj).all()
            else:
                assert d[name] == obj

