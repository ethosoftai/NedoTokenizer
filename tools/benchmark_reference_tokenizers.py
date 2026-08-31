#!/usr/bin/env python3
import argparse, json, struct, time
from collections import defaultdict
from tokenizers import Tokenizer
import tiktoken
MAGIC=b'MVOCBIN1'

def iter_records(path):
    with open(path,'rb',buffering=8<<20) as f:
        assert f.read(8)==MAGIC
        while True:
            h=f.read(13)
            if not h:return
            sid=h[0];n=struct.unpack('<I',h[9:13])[0];raw=f.read(n)
            if len(raw)!=n:raise RuntimeError('truncated')
            yield sid,raw

def main():
    p=argparse.ArgumentParser();p.add_argument('--eval',required=True);p.add_argument('--raw-bpe-json');p.add_argument('--output',required=True)
    a=p.parse_args(); refs={}
    if a.raw_bpe_json:
        hf=Tokenizer.from_file(a.raw_bpe_json);refs['raw_bpe_32k']=lambda s:len(hf.encode(s).ids)
    for name in ['cl100k_base','o200k_base']:
        enc=tiktoken.get_encoding(name);refs[name]=lambda s,e=enc:len(e.encode_ordinary(s))
    stats={name:{'records':0,'bytes':0,'tokens':0,'invalid_utf8_records':0,'sources':defaultdict(lambda:{'records':0,'bytes':0,'tokens':0})} for name in refs}
    for sid,raw in iter_records(a.eval):
        try:text=raw.decode('utf8')
        except UnicodeDecodeError:
            for st in stats.values():st['invalid_utf8_records']+=1
            continue
        for name,fn in refs.items():
            n=fn(text);st=stats[name];st['records']+=1;st['bytes']+=len(raw);st['tokens']+=n
            ss=st['sources'][str(sid)];ss['records']+=1;ss['bytes']+=len(raw);ss['tokens']+=n
    for st in stats.values():
        st['bytes_per_token']=st['bytes']/max(1,st['tokens'])
        st['sources']=dict(st['sources'])
        for ss in st['sources'].values():ss['bytes_per_token']=ss['bytes']/max(1,ss['tokens'])
    open(a.output,'w').write(json.dumps({'status':'PASS','results':stats},indent=2,sort_keys=True)+'\n')
    for name,st in stats.items():print(name,'tokens',st['tokens'],'bytes/token',st['bytes_per_token'])
if __name__=='__main__':main()
