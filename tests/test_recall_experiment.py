import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('recall_train',Path(__file__).resolve().parents[1]/'experiments/pii_recall/train.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class Tokenizer:
    def __call__(self,*args,**kwargs):
        return {'input_ids':[[1,2,3,4,5]],'offset_mapping':[[(0,0),(0,4),(5,9),(10,16),(0,0)]]}

def test_nested_partial_and_multiword_bio():
    row={'text':'Иван Петр 123456','annotated_labels':['PERSON','PASSPORT','PASSPORT_NUMBER'],
         'entities':[{'start':0,'end':9,'label':'PERSON'},{'start':5,'end':16,'label':'PASSPORT'},
                     {'start':10,'end':16,'label':'PASSPORT_NUMBER'}]}
    y=m.encode(row,Tokenizer(),32,8)[0]['targets']
    assert y[1:3,m.TYPES.index('PERSON')].tolist()==[1,2]
    assert y[2:4,m.TYPES.index('PASSPORT')].tolist()==[1,2]
    assert y[3,m.TYPES.index('PASSPORT_NUMBER')]==1
    assert (y[:,m.TYPES.index('EMAIL')]==-100).all()
    assert (y[0]==-100).all() and (y[-1]==-100).all()

def test_decode_recovers_orphan_i_and_keeps_nested():
    scores=np.zeros((3,len(m.TYPES),3)); scores[:,:,0]=1
    scores[:,m.TYPES.index('PASSPORT'),2]=3
    scores[2,m.TYPES.index('PASSPORT_NUMBER'),1]=3
    got=m.decode([(0,2),(3,5),(6,12)],scores)
    assert {'start':0,'end':12,'label':'PASSPORT'} in got
    assert {'start':6,'end':12,'label':'PASSPORT_NUMBER'} in got

def test_split_leakage():
    with pytest.raises(ValueError,match='leakage'):
        m.check_splits({'train':[{'text':'ABC  def'}],'test':[{'text':'abc def'}]})

def test_missing_required_label_cannot_pass():
    row={'text':'Иван','annotated_labels':['PERSON'],'entities':[{'start':0,'end':4,'label':'PERSON'}]}
    scores=np.zeros((1,len(m.TYPES),3)); scores[:,:,0]=1; scores[0,m.TYPES.index('PERSON'),1]=2
    result=m.report([row],[([(0,4)],scores)],0,.95)
    assert result['exact_span_micro']['recall']==1
    assert not result['required_labels_all_pass']

def test_window_inference_retains_tail_and_prefers_center():
    import torch
    from types import SimpleNamespace
    class Windows:
        pad_token_id=0
        def __call__(self,*args,**kwargs):
            return {'input_ids':[[0,1,2,3,0],[0,3,4,5,0]],
                    'offset_mapping':[[(0,0),(0,1),(2,3),(4,5),(0,0)],
                                      [(0,0),(4,5),(6,7),(8,9),(0,0)]]}
    class Model:
        def eval(self): pass
        def __call__(self,ids,mask):
            scores=torch.zeros((*ids.shape,len(m.TYPES),3))
            scores[:,:,:,0]=1
            scores[:,:,m.TYPES.index('PERSON'),1]=(ids==5)*3
            return scores
    row={'text':'a b c d e','annotated_labels':['PERSON'],'entities':[]}
    output=m.collect_logits(Model(),[row],Windows(),SimpleNamespace(max_length=5,stride=1,batch_size=1,device='cpu'))
    assert output[0][0]==[(0,1),(2,3),(4,5),(6,7),(8,9)]
    assert {'start':8,'end':9,'label':'PERSON'} in m.decode(*output[0])

def test_bert_1024_extension_trains_and_roundtrips(tmp_path):
    import torch
    from transformers import BertConfig, BertModel, AutoModel
    torch.set_num_threads(1)
    encoder = BertModel(BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=32,
                                  max_position_embeddings=512, hidden_dropout_prob=0,
                                  attention_probs_dropout_prob=0))
    original = encoder.embeddings.position_embeddings.weight.detach().clone()
    assert m.extend_bert_positions(encoder, 1024) == 512
    torch.testing.assert_close(encoder.embeddings.position_embeddings.weight[:512], original)
    assert encoder.embeddings.position_ids.shape == (1,1024)
    assert encoder.embeddings.token_type_ids.shape == (1,1024)
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    encoder.train()
    ids = torch.ones((1,1024),dtype=torch.long)
    output = encoder(input_ids=ids).last_hidden_state
    output[0,900,0].backward()
    assert encoder.embeddings.position_embeddings.weight.grad[900].abs().sum() > 0
    encoder.eval()
    with torch.no_grad(): expected=encoder(input_ids=ids).last_hidden_state
    encoder.save_pretrained(tmp_path)
    loaded=AutoModel.from_pretrained(tmp_path,local_files_only=True).eval()
    with torch.no_grad(): actual=loaded(input_ids=ids).last_hidden_state
    torch.testing.assert_close(actual,expected)
    assert loaded.config.max_position_embeddings == 1024


def test_position_extension_does_not_shrink():
    from transformers import BertConfig, BertModel
    encoder=BertModel(BertConfig(vocab_size=8,hidden_size=8,num_hidden_layers=1,
                                num_attention_heads=1,intermediate_size=16,max_position_embeddings=512))
    assert m.extend_bert_positions(encoder,384)==512
    assert encoder.embeddings.position_embeddings.num_embeddings==512

def test_reuse_dataset_bundles_current_code_without_corpus(tmp_path):
    import ast
    import json
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'scripts/push_kaggle_recall.py'),
                    '--model','base','--max-length','1024','--extend-positions',
                    '--reuse-dataset','owner/existing-private-corpus','--out',str(tmp_path)],check=True)
    assert not (tmp_path/'dataset').exists()
    assert not list(tmp_path.rglob('*.jsonl'))
    metadata=json.loads((tmp_path/'kernel-metadata.json').read_text())
    assert metadata['dataset_sources']==['owner/existing-private-corpus']
    assert metadata['is_private'] is True
    module=ast.parse((tmp_path/'kaggle_recall_train.py').read_text())
    config=ast.literal_eval(module.body[0].value)
    sources=ast.literal_eval(module.body[1].value)
    assert config['max_length']==1024 and config['extend_positions']
    assert sources['experiments/pii_recall/train.py']==(root/'experiments/pii_recall/train.py').read_text()
