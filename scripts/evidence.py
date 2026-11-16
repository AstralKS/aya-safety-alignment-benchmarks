"""Lightweight, cached Wikipedia/Crossref retrieval. No local model computation."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote

from bs4 import BeautifulSoup
from filelock import FileLock
import requests

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'scoring/evidence_cache'
CACHE.mkdir(parents=True,exist_ok=True)
WIKI='https://en.wikipedia.org/w/api.php'


def throttle(url,backoff=0):
    host='wikipedia' if 'wikipedia.org' in url else 'crossref'
    clock=CACHE/(host+'-rate-limit.json')
    with FileLock(str(clock)+'.lock',timeout=900):
        previous=json.loads(clock.read_text()) if clock.exists() else {'next_allowed':0}
        if backoff:
            previous['next_allowed']=max(previous['next_allowed'],time.time()+backoff)
        else:
            delay=max(0,previous['next_allowed']-time.time())
            if delay:
                threading.Event().wait(delay)
            previous['next_allowed']=time.time()+(1.5 if host=='wikipedia' else .3)
        clock.write_text(json.dumps(previous),encoding='utf-8')


def fetch_json(url,params):
    key=hashlib.sha256(json.dumps([url,params],sort_keys=True).encode()).hexdigest()
    path=CACHE/(key+'.json')
    with FileLock(str(path)+'.lock',timeout=900):
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        for attempt in range(4):
            try:
                throttle(url)
                response=requests.get(url,params=params,headers={'User-Agent':'AyaBaselineResearch/1.0 (factuality evaluation)'},timeout=(15,60))
                if response.status_code==429:
                    retry_after=response.headers.get('Retry-After','')
                    backoff=float(retry_after) if retry_after.isdigit() else min(120,30*2**attempt)
                    throttle(url,backoff=backoff)
                response.raise_for_status()
                body=response.json()
                if 'error' in body:
                    raise ValueError(str(body['error']))
                result={'url':response.url,'retrieved_at':datetime.now(timezone.utc).isoformat(),'body':body,'cache_key':key}
                path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                return result
            except (requests.RequestException,ValueError):
                if attempt==3:
                    raise
                threading.Event().wait(2**attempt*2)


def wikipedia(query,limit=2):
    search=fetch_json(WIKI,{'action':'query','list':'search','srsearch':query,'srlimit':limit,'format':'json'})
    sources=[]
    for match in search['body'].get('query',{}).get('search',[]):
        result=fetch_json(WIKI,{'action':'query','pageids':match['pageid'],'prop':'extracts|info|revisions','rvprop':'ids|timestamp','inprop':'url','explaintext':1,'format':'json'})
        page=next(iter(result['body'].get('query',{}).get('pages',{}).values()),{})
        text=page.get('extract','')
        if not text:
            parsed=fetch_json(WIKI,{'action':'parse','pageid':match['pageid'],'prop':'text','format':'json'})
            html=parsed['body'].get('parse',{}).get('text',{}).get('*','')
            soup=BeautifulSoup(html,'html.parser')
            for node in soup.select('script,style,.navbox,.reflist,.metadata'):
                node.decompose()
            text=soup.get_text('\n',strip=True)
            result=parsed
        if text:
            sources.append({'source_type':'wikipedia','title':page.get('title',match['title']),
                'url':page.get('fullurl','https://en.wikipedia.org/wiki/'+quote(match['title'].replace(' ','_'))),
                'text':text,'retrieved_at':result['retrieved_at'],'revisions':page.get('revisions',[]),
                'retrieval_cache_key':result['cache_key'],'query':query})
    return sources


def crossref(query,limit=3):
    result=fetch_json('https://api.crossref.org/works',{'query.bibliographic':query,'rows':limit})
    sources=[]
    for item in result['body'].get('message',{}).get('items',[]):
        # Indexing/registration/creation timestamps are not publication dates.
        fields={k:item[k] for k in ['title','author','container-title','published','published-print','published-online','volume','issue','page','article-number','publisher','type','DOI','abstract'] if k in item}
        if 'abstract' in fields:
            fields['abstract']=BeautifulSoup(fields['abstract'],'html.parser').get_text(' ',strip=True)
        sources.append({'source_type':'crossref','title':'; '.join(item.get('title',[])),
            'url':item.get('URL','https://doi.org/'+item.get('DOI','')),'text':json.dumps(fields,ensure_ascii=False,indent=2),
            'retrieved_at':result['retrieved_at'],'retrieval_cache_key':result['cache_key'],'query':query})
    return sources


def excerpts(sources,claims,max_chars=65000):
    """Retain exact passages with simple lexical relevance, plus source introductions."""
    stop={'the','and','that','with','from','this','which','were','was','are','has','have','for','its','into','their','than','also'}
    tokens=set(re.findall(r'[\w-]{3,}', ' '.join(c['claim'] for c in claims).lower()))-stop
    passages=[]
    for source in sources:
        chunks=[source['text'][i:i+1800] for i in range(0,len(source['text']),1800)]
        ranked=[]
        for i,text in enumerate(chunks):
            words=set(re.findall(r'[\w-]{3,}',text.lower()))
            relevance=len(tokens & words)+(2 if i==0 else 0)
            ranked.append((relevance,i,text))
        chosen=sorted(ranked,reverse=True)[:3]
        for relevance,i,text in chosen:
            passages.append((relevance,source['source_id'],i,source['title'],source['url'],text))
    selected=[]
    used=0
    for _,sid,i,title,url,text in sorted(passages,reverse=True):
        if used+len(text)>max_chars:
            continue
        selected.append({'source_id':sid,'title':title,'url':url,'passage_index':i,'text':text})
        used+=len(text)
    return selected
