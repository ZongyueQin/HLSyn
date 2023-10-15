import torch
from utils import get_root_path
from glob import iglob
from os.path import join

models_path = [f for f in iglob(join(get_root_path(), 'save/programl/**'), recursive = True) if f.endswith('.pth') and 'node-att' in f]

for model in models_path:
    loaded_model = torch.load(model, map_location=torch.device('cpu'))
    #loaded_model.eval()
    print(model)
    for param in loaded_model:
        print(param)

    print()
    print('##########################################################')
    print()
    # break