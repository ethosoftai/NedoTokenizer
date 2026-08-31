#!/usr/bin/env python3
import argparse, hashlib, json, os, struct, time
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import tokenizers as tokenizers_pkg

MAGIC=b'MVOCBIN1'

def splitmix64(v):
    v=(v+0x9e3779b97f4a7c15)&0xffffffffffffffff
    v=((v^(v>>30))*0xbf58476d1ce4e5b9)&0xffffffffffffffff
    v=((v^(v>>27))*0x94d049bb133111eb)&0xffffffffffffffff
    return v^(v>>31)

def records(path, seed, mod):
    with open(path,'rb',buffering=8<<20) as f:
        assert f.read(8)==MAGIC
        while True:
            h=f.read(13)
            if not h:return
            if len(h)!=13:raise RuntimeError('truncated header')
            key=struct.unpack('<Q',h[1:9])[0]; n=struct.unpack('<I',h[9:13])[0]
            raw=f.read(n)
            if len(raw)!=n:raise RuntimeError('truncated text')
            if splitmix64(key^seed)%mod==0:continue
            try: yield raw.decode('utf8')
            except UnicodeDecodeError: continue

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--total-vocab',type=int,default=32000);p.add_argument('--eval-modulus',type=int,default=20)
    p.add_argument('--split-seed',type=lambda x:int(x,0),default=0x4e45444f42504531);p.add_argument('--max-token-bytes',type=int,default=96)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);t0=time.time()
    tok=Tokenizer(models.BPE(unk_token=None));tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=True);tok.decoder=decoders.ByteLevel()
    tr=trainers.BpeTrainer(vocab_size=a.total_vocab-3,min_frequency=2,show_progress=True,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),max_token_length=a.max_token_bytes)
    tok.train_from_iterator(records(a.input,a.split_seed,a.eval_modulus),trainer=tr)
    tok.save(str(out/'raw-gpt2-byte-bpe-tokenizer.json')); tok.model.save(str(out),'raw-gpt2-byte-bpe')
    m={'schema':'raw_bpe_baseline_v1','status':'PASS','input':a.input,'total_model_vocab_equivalent':a.total_vocab,'tokenizer_vocab':tok.get_vocab_size(),'eval_modulus':a.eval_modulus,'split_seed':a.split_seed,'tokenizers_version':tokenizers_pkg.__version__,'elapsed_seconds':time.time()-t0}
    (out/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    print(json.dumps(m,indent=2))
if __name__=='__main__':main()
